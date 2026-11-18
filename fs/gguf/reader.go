package gguf

import (
	"bufio"
	"encoding/binary"
	"io"
)

type bufferedReader struct {
	offset int64
	word   [8]byte
	*bufio.Reader
}

func newBufferedReader(rs io.ReadSeeker, size int) *bufferedReader {
	return &bufferedReader{
		Reader: bufio.NewReaderSize(rs, size),
	}
}

func (rs *bufferedReader) Read(p []byte) (n int, err error) {
	n, err = rs.Reader.Read(p)
	rs.offset += int64(n)
	return n, err
}

func (rs *bufferedReader) uint64(order binary.ByteOrder) (uint64, error) {
	if _, err := io.ReadFull(rs, rs.word[:]); err != nil {
		return 0, err
	}
	return order.Uint64(rs.word[:]), nil
}

func (rs *bufferedReader) discard(n int) error {
	read, err := rs.Reader.Discard(n)
	rs.offset += int64(read)
	return err
}
