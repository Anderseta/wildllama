package server

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestImageJobTimingsAccumulateAcrossImagesAndFreezeAtCompletion(t *testing.T) {
	created := time.Unix(100, 0)
	at := func(seconds int) time.Time { return created.Add(time.Duration(seconds) * time.Second) }
	started := at(3)
	job := &imageJob{CreatedAt: created, StartedAt: &started}
	job.startImageTiming(started)
	job.trackImagePhase("loading", at(5))
	job.trackImagePhase("sampling", at(8))
	job.trackImagePhase("sampling", at(9))
	job.trackImagePhase("decoding", at(11))
	job.trackImagePhase("completed", at(12))
	job.trackImagePhase("loading", at(13))
	job.trackImagePhase("sampling", at(15))

	live := job.imageTimingSnapshot(at(18))
	require.Equal(t, float64(3), live.QueueSeconds)
	require.Equal(t, float64(15), live.RunSeconds)
	require.Equal(t, float64(18), live.TotalSeconds)
	require.Equal(t, float64(5), *live.FirstSamplingSeconds)
	require.Equal(t, map[string]float64{"preparing": 2, "loading": 5, "sampling": 6, "decoding": 1, "publishing": 1}, live.PhaseSeconds)
	live.PhaseSeconds["sampling"] = 99
	*live.FirstSamplingSeconds = 99

	job.trackImagePhase("restoring_text", at(19))
	job.finishImageTiming(at(24))
	completed := job.imageTimingSnapshot(at(100))
	require.Equal(t, float64(24), completed.TotalSeconds)
	require.Equal(t, float64(5), *completed.FirstSamplingSeconds)
	require.Equal(t, map[string]float64{"preparing": 2, "loading": 5, "sampling": 7, "decoding": 1, "publishing": 1, "restoring_text": 5}, completed.PhaseSeconds)
}

func TestImageJobSnapshotsDetachPeakMemory(t *testing.T) {
	service := &imageService{}
	job := &imageJob{}
	service.updateImageProgress(job, "loading", "", 0, 0, 2048)
	snapshot := cloneImageJobLocked(job)
	service.updateImageProgress(job, "sampling", "", 0, 0, 4096)
	service.updateImageProgress(job, "decoding", "", 0, 0, 1024)
	require.Equal(t, uint64(2048), snapshot.Memory.PeakRSSBytes)
	require.Equal(t, uint64(4096), job.Memory.PeakRSSBytes)
}
