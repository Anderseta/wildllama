package server

import (
	"maps"
	"time"
)

type imageJobTimings struct {
	QueueSeconds         float64            `json:"queue_seconds"`
	RunSeconds           float64            `json:"run_seconds"`
	TotalSeconds         float64            `json:"total_seconds"`
	FirstSamplingSeconds *float64           `json:"first_sampling_seconds,omitempty"`
	PhaseSeconds         map[string]float64 `json:"phase_seconds,omitempty"`
}

type imageJobMemory struct {
	PeakRSSBytes uint64 `json:"peak_rss_bytes,omitempty"`
}

// The service lock protects the active phase and its timestamps. Inactive
// phases accumulate across retries and batch images.
func (job *imageJob) trackImagePhase(phase string, now time.Time) {
	if job.Timings == nil || job.timingPhaseStarted.IsZero() {
		return
	}
	if (phase == "sampling" || phase == "generation") && job.Timings.FirstSamplingSeconds == nil {
		seconds := max(0, now.Sub(*job.StartedAt).Seconds())
		job.Timings.FirstSamplingSeconds = &seconds
	}
	// A runner completion describes the image. The job still publishes the
	// asset and can have more images or text restoration remaining.
	if phase == "completed" {
		phase = "publishing"
	}
	if phase == job.timingPhase {
		return
	}
	job.Timings.PhaseSeconds[job.timingPhase] += max(0, now.Sub(job.timingPhaseStarted).Seconds())
	job.timingPhase, job.timingPhaseStarted = phase, now
}

func (job *imageJob) imageTimingSnapshot(now time.Time) *imageJobTimings {
	if job.Timings == nil {
		return nil
	}
	result := *job.Timings
	result.PhaseSeconds = maps.Clone(result.PhaseSeconds)
	if result.FirstSamplingSeconds != nil {
		value := *result.FirstSamplingSeconds
		result.FirstSamplingSeconds = &value
	}
	if !job.timingPhaseStarted.IsZero() {
		result.RunSeconds = max(0, now.Sub(*job.StartedAt).Seconds())
		result.TotalSeconds = result.QueueSeconds + result.RunSeconds
		result.PhaseSeconds[job.timingPhase] += max(0, now.Sub(job.timingPhaseStarted).Seconds())
	}
	return &result
}

func (job *imageJob) startImageTiming(now time.Time) {
	job.Timings = &imageJobTimings{
		QueueSeconds: max(0, now.Sub(job.CreatedAt).Seconds()),
		PhaseSeconds: make(map[string]float64),
	}
	job.timingPhase, job.timingPhaseStarted = "preparing", now
}

func (job *imageJob) finishImageTiming(now time.Time) {
	job.Timings = job.imageTimingSnapshot(now)
	job.timingPhase, job.timingPhaseStarted = "", time.Time{}
}
