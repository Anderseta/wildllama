package server

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestImageRAMBudgetAPIRevisionAndRestart(t *testing.T) {
	_, service, router, _, _ := newImageTestService(t)
	config := cloneImageConfig(service.config)
	config.Profiles[0].MaxRAMMiB = 15360
	config.Profiles[0].MaxVRAMMiB = 12288
	config.Profiles[0].VRAMReserveMiB = 0
	config.Profiles[0].Name = "Custom profile · imagem"
	config.Profiles[0].PromptPrefix = "Prefix with exact spacing  "
	config.Profiles[0].NegativePrompt = "No extra text"
	config.Profiles[0].ExecutionMode = "auto"
	seed := int64(0)
	config.Profiles[0].Seed = &seed
	before := config.Profiles[0]
	data, err := json.Marshal(config)
	require.NoError(t, err)
	response := imageTestHTTP(router, "PATCH", "/api/images/config", data, "application/json")
	require.Equal(t, 200, response.Code, response.Body.String())
	var saved struct {
		Config imageConfig `json:"config"`
	}
	require.NoError(t, json.Unmarshal(response.Body.Bytes(), &saved))
	require.Equal(t, before, saved.Config.Profiles[0])
	t.Setenv("WILDLLAMA_IMAGES_DIR", service.root)
	restarted := &Server{}
	t.Cleanup(restarted.stopImageJobs)
	loaded := restarted.imageService()
	require.Empty(t, loaded.configError)
	require.Equal(t, before, loaded.config.Profiles[0])
	// A queued job owns a copy of both independent resource budgets.
	snapshot := cloneImageConfig(loaded.config)
	loaded.config.Profiles[0].MaxRAMMiB = 2048
	require.Equal(t, 15360, snapshot.Profiles[0].MaxRAMMiB)
	require.Equal(t, 12288, snapshot.Profiles[0].MaxVRAMMiB)
}

func TestImageOffloadAdmissionUsesVRAMCeilingInsteadOfRequiringFullBudgetFree(t *testing.T) {
	for _, scenario := range []struct {
		name               string
		freeMiB, targetMiB int
		wantMiB            int
		cpu                bool
	}{
		{"eight-GiB-card-with-twelve-GiB-cap", 8192, 12288, 7168, false},
		{"explicit-non-GiB-cap", 8192, 1025, 1025, false},
		{"automatic-cap", 8192, 0, 7168, false},
		{"no-headroom", 512, 12288, 12288, true},
		{"insufficient-working-set", 1064, 12288, 12288, true},
		{"explicit-tiny-GPU-cap", 8192, 1, 1, true},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			_, service, _, events, _ := newImageTestService(t)
			service.text = &imageRecoveryText{imageTestText: &imageTestText{events: events}, free: []uint64{uint64(scenario.freeMiB) << 20}}
			profile := service.config.Profiles[0].Profile
			profile.MaxVRAMMiB, profile.MaxRAMMiB = scenario.targetMiB, 15360
			profile.VRAMReserveMiB = 1024
			before := profile
			effective, cpu, free, err := service.admitImageMemory(t.Context(), profile)
			require.NoError(t, err)
			require.Equal(t, scenario.cpu, cpu)
			require.Equal(t, uint64(scenario.freeMiB)<<20, free)
			require.Equal(t, scenario.wantMiB, effective.MaxVRAMMiB)
			require.Equal(t, 15360, effective.MaxRAMMiB)
			require.Equal(t, before, profile, "admission must not change configured model limits")
			require.Equal(t, []string{"free_memory"}, events.snapshot(), "no full-budget wait is needed for offload")
		})
	}
}

func TestImageOffloadAdmissionHonorsCancellationAndExplicitlyDisabledCPU(t *testing.T) {
	_, service, _, events, _ := newImageTestService(t)
	service.text = &imageRecoveryText{imageTestText: &imageTestText{events: events}, free: []uint64{8 << 30}}
	profile := service.config.Profiles[0].Profile
	profile.MaxVRAMMiB = 12288
	profile.VRAMReserveMiB = 1024
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	_, _, _, err := service.admitImageMemory(ctx, profile)
	require.ErrorIs(t, err, context.Canceled)
	require.Empty(t, events.snapshot())
	profile.OffloadToCPU = false
	ctx, cancel = context.WithTimeout(t.Context(), 10*time.Millisecond)
	defer cancel()
	_, cpu, _, err := service.admitImageMemory(ctx, profile)
	require.ErrorIs(t, err, context.DeadlineExceeded)
	require.False(t, cpu, "a disabled CPU fallback must not be silently enabled")
}

func TestImageDefaultMarginBelongsToProfileAndExplicitZeroIsPreserved(t *testing.T) {
	_, service, _, events, _ := newImageTestService(t)
	service.config.ReserveVRAMMiB = 8192 // compatibility field is not the active profile's margin
	service.text = &imageRecoveryText{imageTestText: &imageTestText{events: events}, free: []uint64{8 << 30}}
	profile := service.config.Profiles[0].Profile
	profile.MaxVRAMMiB = 0
	require.Equal(t, 256, profile.VRAMReserveMiB)
	effective, cpu, _, err := service.admitImageMemory(t.Context(), profile)
	require.NoError(t, err)
	require.False(t, cpu)
	require.Equal(t, 8192-256, effective.MaxVRAMMiB)
	profile.VRAMReserveMiB = 0
	encoded, err := json.Marshal(profile)
	require.NoError(t, err)
	var loaded imageProfile
	require.NoError(t, json.Unmarshal(encoded, &loaded))
	require.Zero(t, loaded.VRAMReserveMiB)
	effective, cpu, _, err = service.admitImageMemory(t.Context(), loaded.Profile)
	require.NoError(t, err)
	require.False(t, cpu)
	require.Equal(t, 8192, effective.MaxVRAMMiB)
}

func TestImageAdmissionKeepsSavedProfileAndGenerationLimits(t *testing.T) {
	for _, freeMiB := range []int{8192, 512} {
		t.Run(fmt.Sprint(freeMiB), func(t *testing.T) {
			_, service, _, events, runner := newImageTestService(t)
			service.config.ReserveVRAMMiB = 1024
			service.config.Profiles[0].MaxVRAMMiB = 12288
			service.config.Profiles[0].MaxRAMMiB = 15360
			service.config.Profiles[0].VRAMReserveMiB = 1024
			service.text = &imageRecoveryText{imageTestText: &imageTestText{events: events}, free: []uint64{uint64(freeMiB) << 20}}
			job := imageLifecycleJob(service, "admission")
			before := job.Profile
			require.NoError(t, service.generate(job))
			require.Equal(t, before, job.Profile)
			require.Equal(t, 12288, job.Generation.Profile.MaxVRAMMiB)
			require.Equal(t, 15360, job.Generation.Profile.MaxRAMMiB)
			require.Equal(t, 15360, runner.profile.MaxRAMMiB)
			require.Equal(t, freeMiB < 1024, runner.request.CPUOnly)
			if freeMiB >= 1024 {
				require.Equal(t, freeMiB-1024, runner.profile.MaxVRAMMiB)
			}
			require.Equal(t, []string{"suspend_text", "free_memory", "run_image", "close_image", "free_memory", "restore_text"}, events.snapshot())
		})
	}
}

func TestImageRAMBudgetAPIRejectsInvalidValuesWithoutChangingConfig(t *testing.T) {
	_, service, router, _, _ := newImageTestService(t)
	before := cloneImageConfig(service.config)
	for _, limit := range []int{-1, 1048577} {
		config := cloneImageConfig(before)
		config.Profiles[0].MaxRAMMiB = limit
		data, err := json.Marshal(config)
		require.NoError(t, err)
		response := imageTestHTTP(router, "PATCH", "/api/images/config", data, "application/json")
		require.Equal(t, 400, response.Code, response.Body.String())
		require.Equal(t, before, service.config)
	}
	_, err := os.Stat(filepath.Join(service.root, "config.json"))
	require.True(t, os.IsNotExist(err), "invalid budgets must not write a configuration")
}

func TestImageLegacyRAMDefaultDoesNotRewriteSavedProfile(t *testing.T) {
	root := t.TempDir()
	config := defaultImageConfig(root)
	config.Profiles[0].MaxVRAMMiB = 12288
	config.Profiles[0].OffloadToCPU = true
	config.Profiles[0].StageComponents = true
	data, err := json.Marshal(config)
	require.NoError(t, err)
	var legacy map[string]any
	require.NoError(t, json.Unmarshal(data, &legacy))
	delete(legacy["profiles"].([]any)[0].(map[string]any), "max_ram_mib")
	delete(legacy["profiles"].([]any)[0].(map[string]any), "vram_reserve_mib")
	data, err = json.Marshal(legacy)
	require.NoError(t, err)
	path := filepath.Join(root, "config.json")
	require.NoError(t, os.WriteFile(path, data, 0600))
	t.Setenv("WILDLLAMA_IMAGES_DIR", root)
	server := &Server{}
	t.Cleanup(server.stopImageJobs)
	loaded := server.imageService()
	require.Empty(t, loaded.configError)
	require.Zero(t, loaded.config.Profiles[0].MaxRAMMiB)
	require.Equal(t, config.Profiles[0], loaded.config.Profiles[0])
	after, err := os.ReadFile(path)
	require.NoError(t, err)
	require.Equal(t, data, after)
}

func TestImageComputeBackendSurvivesProgressPollingAndDoesNotLeakToQueuedJob(t *testing.T) {
	_, service, router, _, _ := newImageTestService(t)
	job := imageLifecycleJob(service, "backend-progress")
	job.Status = "running"
	service.jobs = append(service.jobs, job)
	service.setImageComputeBackend(job, false)
	require.Equal(t, "cuda0", job.ComputeBackend)
	require.False(t, job.RAMOffload)
	service.update(job, "ram_offload", "Using RAM weights with GPU compute", 0, 0)
	service.update(job, "sampling", "Sampling with GPU", 1, 25)
	response := imageTestHTTP(router, "GET", "/api/images/jobs/"+job.ID, nil, "")
	var ramCurrent struct {
		Job imageJob `json:"job"`
	}
	require.NoError(t, json.Unmarshal(response.Body.Bytes(), &ramCurrent))
	require.Equal(t, "cuda0", ramCurrent.Job.ComputeBackend)
	require.True(t, ramCurrent.Job.RAMOffload, "RAM/GPU selection must survive a missed transition event")
	service.update(job, "cpu_fallback", "GPU memory failure; retrying on CPU", 0, 0)
	service.update(job, "loading", "Loading model components", 0, 0)
	service.update(job, "sampling", "Sampling", 1, 25)
	response = imageTestHTTP(router, "GET", "/api/images/jobs/"+job.ID, nil, "")
	require.Equal(t, 200, response.Code)
	var current struct {
		Job imageJob `json:"job"`
	}
	require.NoError(t, json.Unmarshal(response.Body.Bytes(), &current))
	require.Equal(t, "sampling", current.Job.Phase)
	require.Equal(t, "cpu", current.Job.ComputeBackend, "a missed short fallback event must not hide CPU execution")
	require.True(t, current.Job.RAMOffload)
	data, err := os.ReadFile(filepath.Join(service.root, "jobs.json"))
	require.NoError(t, err)
	var persisted []imageJob
	require.NoError(t, json.Unmarshal(data, &persisted))
	require.Equal(t, "cpu", persisted[0].ComputeBackend)
	service.workerStarted = true // isolate the queued metadata without starting work
	queued, err := service.enqueue(&imageJob{Kind: "generate", Config: cloneImageConfig(service.config),
		Profile: service.config.Profiles[0], ComputeBackend: "cpu", RAMOffload: true, Request: imageJobRequest{Prompt: "new request"}})
	require.NoError(t, err)
	require.Empty(t, queued.ComputeBackend)
	require.False(t, queued.RAMOffload)
	response = imageTestHTTP(router, "GET", "/api/images/jobs/"+queued.ID, nil, "")
	var public map[string]map[string]any
	require.NoError(t, json.Unmarshal(response.Body.Bytes(), &public))
	_, leaked := public["job"]["compute_backend"]
	require.False(t, leaked, "the previous job's CPU decision must not appear on an unstarted job")
}
