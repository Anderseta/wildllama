package imagegen

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
)

const (
	conditioningDigestMemoName = "conditioning-digests-v1.json"
	conditioningDigestLimit    = 64
	conditioningDigestMaxBytes = 32 << 10
)

type conditioningDigestRecord struct {
	Key    string `json:"key"`
	Digest string `json:"digest"`
}

type conditioningDigestMemo struct {
	root    *os.Root
	entries []conditioningDigestRecord
	dirty   bool
}

type conditioningDigestEnvelope struct {
	Checksum string          `json:"checksum"`
	Entries  json.RawMessage `json:"entries"`
}

func conditioningDigestString(data []byte) string {
	digest := sha256.Sum256(data)
	return hex.EncodeToString(digest[:])
}

func openConditioningDigestMemo(path string) *conditioningDigestMemo {
	root, err := os.OpenRoot(path)
	if err != nil {
		return nil
	}
	memo := &conditioningDigestMemo{root: root}
	info, err := root.Lstat(conditioningDigestMemoName)
	if err != nil || !info.Mode().IsRegular() || info.Size() > conditioningDigestMaxBytes {
		return memo
	}
	file, err := root.Open(conditioningDigestMemoName)
	if err != nil {
		return memo
	}
	defer file.Close()
	if opened, err := file.Stat(); err != nil || !opened.Mode().IsRegular() || !os.SameFile(info, opened) {
		return memo
	}
	data, err := io.ReadAll(io.LimitReader(file, conditioningDigestMaxBytes+1))
	var envelope conditioningDigestEnvelope
	var entries []conditioningDigestRecord
	if err == nil && len(data) <= conditioningDigestMaxBytes && json.Unmarshal(data, &envelope) == nil &&
		conditioningDigestString(envelope.Entries) == envelope.Checksum && json.Unmarshal(envelope.Entries, &entries) == nil && len(entries) <= conditioningDigestLimit {
		memo.entries = entries
	}
	return memo
}

func (memo *conditioningDigestMemo) get(path, stamp string) string {
	if memo == nil {
		return ""
	}
	key := conditioningDigestString([]byte(path + "\x00" + stamp))
	for _, record := range memo.entries {
		if record.Key == key && len(record.Digest) == sha256.Size*2 {
			if _, err := hex.DecodeString(record.Digest); err == nil {
				return record.Digest
			}
		}
	}
	return ""
}

func (memo *conditioningDigestMemo) put(path, stamp, digest string) {
	if memo == nil {
		return
	}
	key := conditioningDigestString([]byte(path + "\x00" + stamp))
	for index, record := range memo.entries {
		if record.Key == key {
			memo.entries[index].Digest = digest
			memo.dirty = true
			return
		}
	}
	memo.entries = append(memo.entries, conditioningDigestRecord{Key: key, Digest: digest})
	if len(memo.entries) > conditioningDigestLimit {
		memo.entries = memo.entries[len(memo.entries)-conditioningDigestLimit:]
	}
	memo.dirty = true
}

func (memo *conditioningDigestMemo) close(ctx context.Context) {
	if memo == nil {
		return
	}
	defer memo.root.Close()
	if !memo.dirty || ctx.Err() != nil {
		return
	}
	entries, err := json.Marshal(memo.entries)
	if err != nil {
		return
	}
	data, err := json.Marshal(conditioningDigestEnvelope{Checksum: conditioningDigestString(entries), Entries: entries})
	if err != nil || CheckDiskSpace(ctx, filepath.Join(memo.root.Name(), conditioningDigestMemoName), int64(len(data))) != nil {
		return
	}
	temporary := ".conditioning-digests-" + rand.Text() + ".tmp"
	file, err := memo.root.OpenFile(temporary, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return
	}
	defer memo.root.Remove(temporary)
	_, writeErr := file.Write(data)
	closeErr := file.Close()
	if writeErr == nil && closeErr == nil {
		// Concurrent writers can lose a cache hit, but every published digest
		// still identifies the exact file stamp it was computed from.
		_ = memo.root.Rename(temporary, conditioningDigestMemoName)
	}
}
