package llm

import (
	"bufio"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

const deltaMagic = "WLDDEL01"
const checkpointDeltaEnv = "WILDLLAMA_CHAT_CHECKPOINT_INCREMENTAL"

func checkpointBase(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	var header [72]byte
	n, err := io.ReadFull(f, header[:])
	if err != nil && !errors.Is(err, io.ErrUnexpectedEOF) && !errors.Is(err, io.EOF) {
		return "", err
	}
	if n < 8 || string(header[:8]) != deltaMagic {
		return "", nil
	}
	if n != len(header) {
		return "", io.ErrUnexpectedEOF
	}
	id := string(header[8:])
	digest, err := hex.DecodeString(id)
	if err != nil || hex.EncodeToString(digest) != id {
		return "", fmt.Errorf("invalid checkpoint base digest")
	}
	return id, nil
}

// Decoding never follows another delta. Cap both the materialized file and
// individual records and reject trailing bytes before submitting to the runner.
func expandCheckpoint(reader io.Reader, base io.ReaderAt, baseSize int64, dst io.Writer, limit int64) (int64, error) {
	r := bufio.NewReaderSize(reader, 64<<10)
	buf := make([]byte, 64<<10)
	var total int64
	for {
		kind, err := r.ReadByte()
		if err != nil {
			return 0, err
		}
		if kind == 0 {
			if _, err = r.ReadByte(); err != io.EOF || total == 0 {
				return 0, fmt.Errorf("invalid checkpoint terminator")
			}
			return total, nil
		}
		if kind != 1 && kind != 2 {
			return 0, fmt.Errorf("invalid checkpoint record")
		}
		var length uint32
		if err = binary.Read(r, binary.LittleEndian, &length); err != nil {
			return 0, err
		}
		n := int64(length)
		if n <= 0 || n > int64(len(buf)) || n > limit-total {
			return 0, fmt.Errorf("invalid checkpoint length")
		}
		chunk := buf[:n]
		if kind == 2 {
			var off uint64
			if err = binary.Read(r, binary.LittleEndian, &off); err != nil {
				return 0, err
			}
			if n > baseSize || off > uint64(baseSize-n) {
				return 0, fmt.Errorf("invalid checkpoint span")
			}
			if _, err = base.ReadAt(chunk, int64(off)); err != nil {
				return 0, err
			}
		} else if _, err = io.ReadFull(r, chunk); err != nil {
			return 0, err
		}
		if _, err = dst.Write(chunk); err != nil {
			return 0, err
		}
		total += n
	}
}

func (c *chatCheckpoint) materialize(source, scratch string, size, scratchBudget int64) (int64, error) {
	digest, err := checkpointBase(source)
	if err != nil {
		return 0, err
	}
	if digest == "" {
		return size, os.Link(source, scratch)
	}
	basePath := filepath.Join(c.directory, c.prefix()+digest+".slot")
	info, err := os.Lstat(basePath)
	if err != nil {
		return 0, err
	}
	if !info.Mode().IsRegular() || info.Size() > c.maxFile {
		return 0, fmt.Errorf("invalid checkpoint base")
	}
	sum, err := handoffChecksum(basePath, info.Size())
	if err != nil || hex.EncodeToString(sum[:]) != digest {
		return 0, fmt.Errorf("damaged checkpoint base")
	}
	base, err := os.Open(basePath)
	if err != nil {
		return 0, err
	}
	defer base.Close()
	delta, err := os.Open(source)
	if err != nil {
		return 0, err
	}
	defer delta.Close()
	if _, err = delta.Seek(72, io.SeekStart); err != nil {
		return 0, err
	}
	dst, err := os.OpenFile(scratch, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return 0, err
	}
	n, writeErr := expandCheckpoint(delta, base, info.Size(), dst, min(c.maxFile, scratchBudget))
	return n, errors.Join(writeErr, dst.Close())
}

// Rotate the keyframe when its last delta grows large. Hard links pin the
// immutable input until the native writer exits, including across chat deletion.
func (c *chatCheckpoint) linkDeltaBase(files []checkpointFile, scratch string) string {
	if os.Getenv(checkpointDeltaEnv) != "1" {
		return ""
	}
	var lastDelta int64
	for _, f := range files {
		if !strings.HasPrefix(f.name, c.prefix()) {
			continue
		}
		path := filepath.Join(c.directory, f.name)
		digest, err := checkpointBase(path)
		if err != nil {
			continue
		}
		if digest != "" {
			lastDelta = f.size
			continue
		}
		if lastDelta > f.size/2 || f.size > c.maxFile {
			return ""
		}
		sum, err := handoffChecksum(path, f.size)
		if err != nil || hex.EncodeToString(sum[:]) != f.name[130:194] {
			return ""
		}
		if err = os.Link(path, scratch+".base"); err != nil {
			return ""
		}
		return f.name[130:194]
	}
	return ""
}

// files is newest first. Evict by the newest member of each snapshot family.
// Within the current family keep the last usable slot and its keyframe. This
// also reclaims superseded files after a crash or a reduction in the quota.
func (c *chatCheckpoint) reserveSaveSpace(files []checkpointFile, used int64) bool {
	target := c.budget - c.maxFile
	if used <= target {
		return true
	}
	own := c.prefix()
	keep, base := "", ""
	groups := make(map[string][]checkpointFile)
	var order []string
	remove := func(f checkpointFile) bool {
		if err := os.Remove(filepath.Join(c.directory, f.name)); err != nil && !errors.Is(err, os.ErrNotExist) {
			return false
		}
		used -= f.size
		return true
	}
	for _, f := range files {
		family := f.name[:130]
		if family != own {
			if len(groups[family]) == 0 {
				order = append(order, family)
			}
			groups[family] = append(groups[family], f)
			continue
		}
		if keep == "" && f.size <= c.maxFile {
			if digest, err := checkpointBase(filepath.Join(c.directory, f.name)); err == nil {
				keep = f.name
				if digest != "" {
					base = own + digest + ".slot"
				}
			}
		}
		if f.name != keep && f.name != base && !remove(f) {
			return false
		}
	}
	for i := len(order) - 1; used > target && i >= 0; i-- {
		for _, f := range groups[order[i]] {
			if !remove(f) {
				return false
			}
		}
	}
	return used <= target
}
