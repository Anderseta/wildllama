package llm

import (
	"bufio"
	"context"
	"io"
	"sync/atomic"
	"time"
)

// Allow normal backend cleanup (callbacks may immediately tokenize), without
// letting a completed SSE response hold its slot until the caller times out.
func closeTerminalStream(ctx context.Context, body io.ReadCloser, scanner *bufio.Scanner) error {
	var expired atomic.Bool
	timer := time.AfterFunc(250*time.Millisecond, func() { expired.Store(true); _ = body.Close() })
	defer timer.Stop()
	read := 0
	for scanner.Scan() {
		read += len(scanner.Bytes())
		if ctx.Err() != nil || read > 64<<10 {
			break
		}
	}
	timer.Stop()
	err := body.Close()
	if ctx.Err() != nil {
		return ctx.Err()
	}
	// The terminal frame already supplied the result. A stalled/truncated tail
	// cannot revoke it; its only remaining purpose was closing the transport.
	if expired.Load() {
		return nil
	}
	return err
}
