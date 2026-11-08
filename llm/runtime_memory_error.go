package llm

import (
	"net/http"
	"strings"

	"github.com/ollama/ollama/api"
)

func (w *StatusWriter) memoryFailureVersion() uint64 {
	if w == nil {
		return 0
	}
	w.memoryErrorMu.Lock()
	defer w.memoryErrorMu.Unlock()
	return w.memoryErrorVersion
}

// A previous request's OOM must not turn a later unrelated error into one.
func (w *StatusWriter) memoryFailureSince(version uint64) string {
	if w == nil {
		return ""
	}
	w.memoryErrorMu.Lock()
	defer w.memoryErrorMu.Unlock()
	if w.memoryErrorVersion <= version {
		return ""
	}
	return w.memoryErrorMessage
}

func (s *llamaServerRunner) runtimeStreamError(message string, since uint64, status int) error {
	if status < 400 {
		status = http.StatusInternalServerError
	}
	detail := s.status.memoryFailureSince(since)
	lower := strings.ToLower(message)
	opaque := strings.Contains(lower, "compute error") || strings.Contains(lower, "without a completion marker") || strings.Contains(lower, "unexpectedly stopped")
	if IsOutOfMemoryMessage(message) || (detail != "" && opaque) {
		if detail != "" && !strings.Contains(message, detail) {
			message += "\n" + detail
		}
		return api.StatusError{StatusCode: http.StatusInsufficientStorage, ErrorMessage: "out of memory: " + message}
	}
	if strings.Contains(lower, "compute error") {
		// A deterministic native compute failure is not a broken connection.
		// Even if its stderr diagnostic is delayed, do not retry the same input.
		return api.StatusError{StatusCode: http.StatusUnprocessableEntity, ErrorMessage: "model compute failed: " + message}
	}
	return api.StatusError{StatusCode: status, ErrorMessage: message}
}

func (s *llamaServerRunner) runtimeStreamBodyError(body []byte, since uint64, status int) error {
	message := extractLlamaServerErrorMessage(body)
	if message == "" {
		message = strings.TrimSpace(string(body))
	}
	return s.runtimeStreamError(message, since, status)
}
