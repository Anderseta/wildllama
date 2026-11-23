package imagegen

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
)

const WanRapidI2V = "wan-rapid-v10-i2v"
const WanRapidT2V = "wan-rapid-v10-t2v"

type VideoOptions struct {
	Frames           int  `json:"frames"`
	FPS              int  `json:"fps"`
	AllowCPUFallback bool `json:"allow_cpu_fallback,omitempty"`
}

func (p Profile) IsWanRapid() bool { return p.ModelID == WanRapidI2V || p.ModelID == WanRapidT2V }

// The Wan pipeline holds an RGB float32 clip on the host for VAE conditioning
// or decode. Weights, spatial tiles, latents and device buffers are additional.
func (v VideoOptions) RGBFloatBytes(width, height int) int64 {
	return int64(width) * int64(height) * int64(v.Frames) * 3 * 4
}

func (v VideoOptions) Validate() error {
	if v.Frames < 5 || v.Frames > 241 || (v.Frames-1)%4 != 0 {
		return errors.New("Wan requires 4n+1 frames between 5 and 241")
	}
	if v.FPS < 1 || v.FPS > 60 {
		return errors.New("video FPS must be between 1 and 60")
	}
	return nil
}

// OutputSpaceBudget uses an uncompressed frame allowance, including room for the container.
func OutputSpaceBudget(width, height int, video *VideoOptions) int64 {
	frames := 1
	if video != nil {
		frames = video.Frames
	}
	return int64(width)*int64(height)*4*int64(frames) + (16 << 20)
}

func validateVideoMode(p Profile, r Request) error {
	if r.Video == nil && !p.IsWanRapid() {
		return nil
	}
	if r.Video == nil || !p.IsWanRapid() || r.Inpaint != nil {
		return errors.New("incompatible video pipeline or request")
	}
	refs := 0
	if p.ModelID == WanRapidI2V {
		refs = 1
	}
	if len(r.ReferenceImages) != refs {
		return fmt.Errorf("this video profile requires %d reference images", refs)
	}
	return r.Video.Validate()
}

// Skip encoded payloads with Seek. The native writer produces one video track
// with one non-laced SimpleBlock per frame. Validate the container before rename.
func webmVint(r io.ByteReader, id bool) (int64, bool, error) {
	b, err := r.ReadByte()
	if err != nil || b == 0 {
		return 0, false, io.ErrUnexpectedEOF
	}
	mask, width := byte(128), 1
	for b&mask == 0 {
		mask >>= 1
		width++
	}
	if id && width > 4 {
		return 0, false, errors.New("invalid EBML element ID")
	}
	n := int64(b)
	if !id {
		n = int64(b & (mask - 1))
	}
	for i := 1; i < width; i++ {
		b, err = r.ReadByte()
		if err != nil {
			return 0, false, err
		}
		n = n<<8 | int64(b)
	}
	return n, !id && n == (1<<(7*width))-1, nil
}

func checkWebM(path string, frames, width, height int) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	return scanWanWebM(file, frames, width, height, nil, nil)
}

// Wan writes independently decodable VP8 keyframes at constant FPS. Rewrap
// the compressed center frame as WebP without decoding or transcoding it.
func MiddleVideoWebP(file *os.File, frames int) ([]byte, error) {
	if frames < 1 || frames > 241 {
		return nil, errors.New("invalid frame count")
	}
	var poster []byte
	err := scanWanWebM(file, frames, 0, 0, func(offset, size int64) error {
		if size < 10 || size > 32<<20 {
			return errors.New("invalid VP8 frame size")
		}
		poster = make([]byte, 20+size+(size&1))
		copy(poster, "RIFF")
		binary.LittleEndian.PutUint32(poster[4:], uint32(len(poster)-8))
		copy(poster[8:], "WEBPVP8 ")
		binary.LittleEndian.PutUint32(poster[16:], uint32(size))
		if _, err := file.ReadAt(poster[20:20+size], offset); err != nil {
			return err
		}
		if poster[20]&1 != 0 || string(poster[23:26]) != "\x9d\x01\x2a" {
			return errors.New("middle frame is not a VP8 keyframe")
		}
		return nil
	}, nil)
	if err == nil && len(poster) == 0 {
		err = errors.New("middle frame unavailable")
	}
	return poster, err
}

// Read the native writer's COMMENT only for historical records without job metadata.
func EmbeddedVideoParameters(file *os.File, frames int) ([]byte, error) {
	var data []byte
	err := scanWanWebM(file, frames, 0, 0, nil, &data)
	return data, err
}

func scanWanWebM(file *os.File, frames, width, height int, middle func(int64, int64) error, comment *[]byte) error {
	info, err := file.Stat()
	if err != nil {
		return err
	}
	var header bool
	var pixelWidth, pixelHeight int
	count := 0
	section := io.NewSectionReader(file, 0, info.Size())
	r := bufio.NewReader(section)
	// Keep the read-ahead for adjacent EBML headers; seek over large blocks.
	position := func() int64 { n, _ := section.Seek(0, io.SeekCurrent); return n - int64(r.Buffered()) }
	var walk func(int64, int64, int64) error
	walk = func(start, end, parent int64) error {
		var tagName string
		var tagText []byte
		for start < end {
			delta := start - position()
			if delta >= 0 && delta <= int64(r.Buffered()) {
				_, _ = r.Discard(int(delta))
			} else {
				if _, err := section.Seek(start, io.SeekStart); err != nil {
					return err
				}
				r.Reset(section)
			}
			id, _, err := webmVint(r, true)
			if err != nil {
				return err
			}
			size, unknown, err := webmVint(r, false)
			if err != nil {
				return err
			}
			pos := position()
			if unknown && id != 0x18538067 {
				return errors.New("unsupported unknown WebM element length")
			}
			if unknown {
				size = end - pos
			}
			if pos > end || size > end-pos {
				return io.ErrUnexpectedEOF
			}
			switch {
			case parent == 0 && start == 0 && id == 0x1a45dfa3:
				header = true
			case parent == 0 && id == 0x18538067 && header,
				parent == 0x18538067 && (id == 0x1f43b675 || id == 0x1654ae6b),
				parent == 0x1654ae6b && id == 0xae, parent == 0xae && id == 0xe0:
				if err := walk(pos, pos+size, id); err != nil {
					return err
				}
			case comment != nil && (parent == 0x18538067 && id == 0x1254c367 || parent == 0x1254c367 && id == 0x7373 || parent == 0x7373 && id == 0x67c8):
				if err := walk(pos, pos+size, id); err != nil {
					return err
				}
			case comment != nil && parent == 0x67c8 && (id == 0x45a3 || id == 0x4487) && size > 0 && size <= 1<<20:
				value := make([]byte, size)
				if _, err := io.ReadFull(r, value); err != nil {
					return err
				}
				if id == 0x45a3 {
					tagName = string(value)
				} else {
					tagText = value
				}
			case parent == 0xe0 && (id == 0xb0 || id == 0xba):
				if size < 1 || size > 4 {
					return errors.New("invalid video track dimensions")
				}
				n := 0
				for range size {
					b, err := r.ReadByte()
					if err != nil {
						return err
					}
					n = n<<8 | int(b)
				}
				if id == 0xb0 {
					pixelWidth = n
				} else {
					pixelHeight = n
				}
			case parent == 0x1f43b675 && id == 0xa0:
				if err := walk(pos, pos+size, id); err != nil {
					return err
				}
			case (parent == 0x1f43b675 && id == 0xa3 || parent == 0xa0 && id == 0xa1) && size > 4:
				if middle != nil && count == frames/2 {
					track, _, err := webmVint(r, false)
					if err != nil || track != 1 {
						return errors.New("invalid Wan video track")
					}
					var h [3]byte
					if _, err = io.ReadFull(r, h[:]); err != nil {
						return err
					}
					offset := position()
					if h[2]&6 != 0 || offset > pos+size {
						return errors.New("unsupported laced video frame")
					}
					if err := middle(offset, pos+size-offset); err != nil {
						return err
					}
				}
				count++
			}
			start = pos + size
		}
		if comment != nil && tagName == "COMMENT" && len(tagText) > 0 {
			*comment = tagText
		}
		return nil
	}
	if err = walk(0, info.Size(), 0); err != nil {
		return err
	}
	if width > 0 && (pixelWidth != width || pixelHeight != height) {
		return fmt.Errorf("invalid WebM: expected %dx%d, found %dx%d", width, height, pixelWidth, pixelHeight)
	}
	if count != frames {
		return fmt.Errorf("invalid WebM: expected %d video frames, found %d", frames, count)
	}
	return nil
}

func publishWebM(partial, target string, frames, width, height int) error {
	if err := checkWebM(partial, frames, width, height); err != nil {
		return fmt.Errorf("incomplete generated video: %w", err)
	}
	return os.Rename(partial, target)
}
