package llm

import (
	"time"

	"github.com/ollama/ollama/api"
)

// GenerationProgressEmitter bounds progress while real output is withheld by a
// parser. Call Next only for a nonempty newly received fragment, never a timer or
// an upstream heartbeat: a genuinely stalled runner must still time out.
// Each request owns its emitter and calls it from its response callback.
type GenerationProgressEmitter struct {
	last time.Time
}

// Stay well inside the usual 180-second client inactivity timeout without
// adding a response callback or network flush for each generated token.
const generationProgressInterval = 30 * time.Second

func (e *GenerationProgressEmitter) Next(phase string) *api.GenerationProgress {
	now := time.Now()
	if !e.last.IsZero() && now.Sub(e.last) < generationProgressInterval {
		return nil
	}
	e.last = now
	return &api.GenerationProgress{Phase: phase}
}
