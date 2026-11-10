package llm

import (
	"bytes"
	"math"
	"strconv"
	"strings"
)

// Native projector lifecycle events contain no user prompt or image data. Keep
// their state separate from the target KV context and discard freed clip buffers.
func (w *memoryParsingWriter) observeProjectorMemoryLocked(line []byte) bool {
	const marker = "wildllama_mmproj "
	if !bytes.HasPrefix(line, []byte(marker)) {
		return false
	}
	fields := strings.Fields(string(line[len(marker):]))
	if len(fields) == 0 || !strings.HasPrefix(fields[0], "state=") {
		return true
	}
	state := strings.TrimPrefix(fields[0], "state=")
	switch state {
	case "metadata", "loading", "loaded", "unloaded", "error":
	default:
		return true
	}
	previous := w.projectorState
	w.projectorState, w.projectorProgress = state, nil
	for _, field := range fields[1:] {
		if value, found := strings.CutPrefix(field, "progress="); found {
			if progress, err := strconv.ParseFloat(value, 64); err == nil && !math.IsNaN(progress) && !math.IsInf(progress, 0) && progress >= 0 && progress <= 1 {
				w.projectorProgress = &progress
			}
		}
	}
	if state == "metadata" || state == "loading" && previous != "loading" || state == "unloaded" || state == "error" {
		for key := range w.buffers {
			if strings.HasPrefix(key.component, "mmproj:") {
				delete(w.buffers, key)
			}
		}
		w.updateRunnerMemoryLocked()
	}
	return true
}
