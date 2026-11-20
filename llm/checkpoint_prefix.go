package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Prepared submissions retain only their wire body. Render that exact body so
// tools, thinking settings and truncated messages match the generation request.
func (s *llamaServerRunner) checkpointPromptTokens(ctx context.Context, wire []byte) ([]int, error) {
	endpoint := fmt.Sprintf("http://127.0.0.1:%d/apply-template", s.port)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(wire))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-WildLlama-Checkpoint-Tokens", "1")
	response, err := s.httpClient().Do(req)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("checkpoint tokenization returned %d", response.StatusCode)
	}
	var rendered struct {
		Tokens []int
		Prompt *string
	}
	if err = json.NewDecoder(response.Body).Decode(&rendered); err != nil {
		return nil, err
	}
	if rendered.Tokens != nil {
		return rendered.Tokens, nil
	}
	if rendered.Prompt == nil {
		return nil, errors.New("checkpoint tokenization unavailable")
	}
	return s.tokenize(ctx, *rendered.Prompt, true)
}

// Inspect only the serialized token prefix. Comparing token IDs directly avoids
// collisions and needs no persistent index or additional user data on disk.
// The native loader still validates and restores every selected state.
func checkpointMatch(input io.Reader, prompt []int) (int, bool, error) {
	r := bufio.NewReaderSize(input, 4096)
	word := func() (uint32, error) {
		b, err := r.Peek(4)
		if err != nil {
			return 0, err
		}
		v := binary.LittleEndian.Uint32(b)
		_, err = r.Discard(4)
		return v, err
	}
	magic, err := word()
	if err != nil {
		return 0, false, err
	}
	version, err := word()
	if err != nil {
		return 0, false, err
	}
	if magic != 0x67677371 || version != 4 {
		return 0, false, errUnknownCheckpointLayout
	}
	packed, err := word()
	if err != nil || packed == 0 {
		return 0, false, io.ErrUnexpectedEOF
	}
	first, err := word()
	if err != nil {
		return 0, false, err
	}
	count, matched := packed, 0
	if first == 0xffffffff {
		format, err := word()
		if err != nil {
			return 0, false, err
		}
		if format != 1 {
			return 0, false, errUnknownCheckpointLayout
		}
		count, err = word()
		if err != nil || packed < 4 || count != packed-4 {
			return 0, false, fmt.Errorf("invalid text checkpoint prefix")
		}
	} else {
		if len(prompt) == 0 || int32(first) < 0 || int(first) != prompt[0] {
			return 0, false, nil
		}
		matched = 1
	}
	for uint64(matched) < uint64(count) && matched < len(prompt) {
		tok, err := word()
		if err != nil {
			return 0, false, err
		}
		if int32(tok) < 0 || int(tok) != prompt[matched] {
			break
		}
		matched++
	}
	return matched, uint64(matched) == uint64(count), nil
}

var errUnknownCheckpointLayout = errors.New("unknown checkpoint token layout")

func (c *chatCheckpoint) match(f checkpointFile, tokens []int) (int, bool, error) {
	path := filepath.Join(c.directory, f.name)
	digest, err := checkpointBase(path)
	if err != nil {
		return 0, false, err
	}
	src, err := os.Open(path)
	if err != nil {
		return 0, false, err
	}
	defer src.Close()
	if digest == "" {
		return checkpointMatch(src, tokens)
	}
	basePath := filepath.Join(c.directory, c.prefix()+digest+".slot")
	info, err := os.Lstat(basePath)
	if err != nil {
		return 0, false, err
	}
	if !info.Mode().IsRegular() || info.Size() > c.maxFile {
		return 0, false, fmt.Errorf("invalid base for checkpoint prefix")
	}
	base, err := os.Open(basePath)
	if err != nil {
		return 0, false, err
	}
	defer base.Close()
	if _, err = src.Seek(72, io.SeekStart); err != nil {
		return 0, false, err
	}
	return checkpointMatch(readCheckpointRecords(src, base, info.Size(), c.maxFile), tokens)
}

func (c *chatCheckpoint) restoreCandidates(all []checkpointFile, prompt func() ([]int, error)) []checkpointFile {
	var files []checkpointFile
	for _, f := range all {
		if strings.HasPrefix(f.name, c.prefix()) && f.size > 0 && f.size <= c.maxFile {
			files = append(files, f)
		}
	}
	if len(files) == 0 || prompt == nil {
		return files
	}
	tokens, err := prompt()
	if err != nil {
		// Older template endpoints may lack text tokenization.
		return files
	}
	type candidate struct {
		file     checkpointFile
		matched  int
		complete bool
	}
	ranked := make([]candidate, 0, len(files))
	// A few shared template tokens rarely justify loading an entire state.
	const minReusableTokens = 256
	for _, f := range files {
		matched, complete, err := c.match(f, tokens)
		if errors.Is(err, errUnknownCheckpointLayout) || (err == nil && matched >= minReusableTokens) {
			ranked = append(ranked, candidate{f, matched, complete})
		}
	}
	sort.SliceStable(ranked, func(i, j int) bool {
		if ranked[i].matched != ranked[j].matched {
			return ranked[i].matched > ranked[j].matched
		}
		return ranked[i].complete && !ranked[j].complete
	})
	files = files[:0]
	for _, r := range ranked {
		files = append(files, r.file)
	}
	return files
}
