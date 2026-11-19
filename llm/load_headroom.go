package llm

import (
	"log/slog"
	"time"

	"github.com/ollama/ollama/ml"
)

// Sample only during loading. Log once per stage, including the last sample on failure.
type loadHeadroom struct {
	last             time.Time
	stage            string
	physical, commit uint64
	samples          int
}

func (h *loadHeadroom) observe(stage string) {
	h.last = time.Now()
	mem, err := ml.ReadWindowsMemory()
	if err != nil || mem.CommitLimit == 0 {
		return
	}
	if stage == "" {
		stage = "startup"
	}
	if h.stage != stage {
		h.report()
		h.stage, h.samples = stage, 0
	}
	if h.samples == 0 || mem.FreeMemory < h.physical {
		h.physical = mem.FreeMemory
	}
	if h.samples == 0 || mem.CommitAvailable < h.commit {
		h.commit = mem.CommitAvailable
	}
	h.samples++
}

func (h *loadHeadroom) report() {
	if h.samples > 0 {
		slog.Info("minimum host headroom during model load", "stage", h.stage, "samples", h.samples,
			"physical_available_mib", h.physical>>20, "commit_available_mib", h.commit>>20)
	}
}
