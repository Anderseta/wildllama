package imagegen

import (
	"bytes"
	"encoding/binary"
	"errors"
	"hash/crc32"
	"io"
	"os"
	"path/filepath"
)

// VideoCheckpoint belongs to one immutable video job. Only its worker may use
// it while running. Save and Close are called after the child has exited.
type VideoCheckpoint struct {
	memory wanReference
	step   int
	size   int64
	seed   int64
}

func (c *VideoCheckpoint) Step() int {
	if c == nil {
		return 0
	}
	return c.step
}
func (c *VideoCheckpoint) Close() {
	if c != nil && c.memory.close != nil {
		c.memory.close()
		*c = VideoCheckpoint{}
	}
}

func WanLatentBytes(width, height, frames int) int64 {
	return int64(width/8) * int64(height/8) * int64(1+(frames-1)/4) * 16 * 4
}

// Read only the publication word after a child exits. Its last log line can be
// lost if it is terminated immediately after publishing a complete latent.
func (m wanReference) publishedStep(limit int) int {
	var n int
	if m.view != nil {
		_ = m.view(4, func(data []byte) error {
			step := binary.LittleEndian.Uint32(data)
			if step > 0 && uint64(step) <= uint64(limit) {
				n = int(step)
			}
			return nil
		})
	}
	return n
}

const checkpointMagic = "WANSTEP1"

type checkpointHeader struct {
	Magic                            [8]byte
	Seed                             int64
	Step, Signature, Bytes, Checksum uint32
}

func (c *VideoCheckpoint) Save(path string) error {
	if c.Step() == 0 || c.memory.view == nil {
		return errors.New("nenhum passo completo para salvar")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		return err
	}
	f, err := os.CreateTemp(filepath.Dir(path), ".wan-step-*")
	if err != nil {
		return err
	}
	defer os.Remove(f.Name())
	err = c.memory.view(int(c.size), func(data []byte) error {
		order := binary.LittleEndian
		n := int(order.Uint32(data))
		if n != c.step {
			return errors.New("passo de vídeo inconsistente")
		}
		size := (len(data) - 16) / 2
		offset := 16 + (n%2)*size
		payload := data[offset : offset+size]
		h := checkpointHeader{Seed: c.seed, Step: uint32(n), Signature: order.Uint32(data[4:]), Bytes: uint32(size)}
		copy(h.Magic[:], checkpointMagic)
		h.Checksum = checkpointCRC(h, payload)
		if err := binary.Write(f, order, h); err != nil {
			return err
		}
		_, err := f.Write(payload)
		return err
	})
	if err == nil {
		err = f.Sync()
	}
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	return os.Rename(f.Name(), path)
}

// Include the seed and schedule metadata in the checksum as well as the payload.
func checkpointCRC(h checkpointHeader, payload []byte) uint32 {
	h.Checksum = 0
	var header bytes.Buffer
	_ = binary.Write(&header, binary.LittleEndian, h)
	return crc32.Update(crc32.ChecksumIEEE(header.Bytes()), crc32.IEEETable, payload)
}

func openVideoCheckpoint(path string, size int64, steps int) (*os.File, checkpointHeader, error) {
	var h checkpointHeader
	f, err := os.Open(path)
	if err != nil {
		return nil, h, err
	}
	info, err := f.Stat()
	if err == nil && (!info.Mode().IsRegular() || size <= 0 || size > 128<<20 || info.Size() != size+int64(binary.Size(h))) {
		err = errors.New("tamanho inválido do passo de vídeo salvo")
	}
	if err == nil {
		err = binary.Read(f, binary.LittleEndian, &h)
	}
	if err == nil && (string(h.Magic[:]) != checkpointMagic || h.Seed < 0 || h.Step == 0 || int64(h.Step) > int64(steps) || int64(h.Bytes) != size) {
		err = errors.New("passo de vídeo incompatível com o perfil original")
	}
	if err != nil {
		f.Close()
		return nil, h, err
	}
	return f, h, nil
}

func SavedVideoStep(path string, size int64, steps int) (int, error) {
	f, h, err := openVideoCheckpoint(path, size, steps)
	if err != nil {
		return 0, err
	}
	return int(h.Step), f.Close()
}

func LoadVideoCheckpoint(path string, size int64, steps int) (*VideoCheckpoint, error) {
	f, h, err := openVideoCheckpoint(path, size, steps)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	mapped := 16 + 2*size
	m := newWanReference(mapped)
	if m.view == nil {
		return nil, errors.New("RAM indisponível para carregar o passo do vídeo")
	}
	err = m.view(int(mapped), func(data []byte) error {
		offset := 16 + int64(h.Step%2)*size
		payload := data[offset : offset+size]
		if _, err := io.ReadFull(f, payload); err != nil {
			return err
		}
		if checkpointCRC(h, payload) != h.Checksum {
			return errors.New("o passo de vídeo salvo está corrompido")
		}
		binary.LittleEndian.PutUint32(data[4:], h.Signature)
		binary.LittleEndian.PutUint32(data, h.Step)
		return nil
	})
	if err != nil {
		m.close()
		return nil, err
	}
	return &VideoCheckpoint{memory: m, step: int(h.Step), size: mapped, seed: h.Seed}, nil
}
