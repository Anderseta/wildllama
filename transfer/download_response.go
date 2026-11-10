package transfer

import (
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"strings"
)

// ErrInvalidDownloadRange means the server did not return the requested blob
// bytes. Retrying the same response cannot safely advance a partial download.
var ErrInvalidDownloadRange = errors.New("invalid blob download range")

// ValidateDownloadResponse checks a blob response before its body reaches disk.
// A full 200 response is allowed only for the entire blob; partial responses
// must identify the exact requested span. Unknown Content-Length (chunked HTTP)
// and unknown complete length in Content-Range remain supported.
func ValidateDownloadResponse(resp *http.Response, offset, size, total int64) error {
	invalid := func(reason string) error { return fmt.Errorf("%w: %s", ErrInvalidDownloadRange, reason) }
	if resp == nil || offset < 0 || size < 0 || total < 0 || offset > total || size > total-offset {
		return invalid("invalid expected byte range")
	}
	switch resp.StatusCode {
	case http.StatusOK:
		if offset != 0 || size != total {
			return invalid("server ignored the requested byte range")
		}
	case http.StatusPartialContent:
		fields := strings.Fields(resp.Header.Get("Content-Range"))
		if size == 0 || len(fields) != 2 || !strings.EqualFold(fields[0], "bytes") {
			return invalid("missing or malformed Content-Range")
		}
		span, complete, ok := strings.Cut(fields[1], "/")
		if !ok {
			return invalid("malformed Content-Range")
		}
		first, last, ok := strings.Cut(span, "-")
		start, firstErr := strconv.ParseUint(first, 10, 63)
		end, lastErr := strconv.ParseUint(last, 10, 63)
		if !ok || firstErr != nil || lastErr != nil || int64(start) != offset || int64(end) != offset+size-1 {
			return invalid("Content-Range does not match the requested bytes")
		}
		if complete != "*" {
			length, err := strconv.ParseUint(complete, 10, 63)
			if err != nil || int64(length) != total {
				return invalid("Content-Range does not match the expected blob size")
			}
		}
	default:
		return fmt.Errorf("unexpected blob download status %d", resp.StatusCode)
	}
	if resp.ContentLength >= 0 && resp.ContentLength != size {
		return invalid("Content-Length does not match the requested bytes")
	}
	return nil
}
