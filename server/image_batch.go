package server

import (
	"errors"
	"fmt"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/imagegen"
)

type imageGenerationSummary struct {
	UserPrompt         string       `json:"user_prompt"`
	UserNegativePrompt string       `json:"user_negative_prompt"`
	Prompt             string       `json:"prompt"`
	NegativePrompt     string       `json:"negative_prompt"`
	ProfileID          string       `json:"profile_id"`
	Profile            imageProfile `json:"profile"`
	ReferenceAssetIDs  []string     `json:"reference_asset_ids"`
	Count              int          `json:"count"`
}

func imageBatchCount(request imageJobRequest) (int, error) {
	count := request.Count
	if count == 0 {
		count = 1
	}
	if count < 1 || count > 8 {
		return 0, errors.New("image batch count must be between 1 and 8")
	}
	return count, nil
}

func imageBatchSeed(request imageJobRequest, profile imageProfile, count int) (int64, error) {
	seed := int64(-1)
	if request.Seed != nil {
		seed = *request.Seed
	} else if profile.Seed != nil {
		seed = *profile.Seed
	}
	if seed < -1 {
		return 0, errors.New("image seed must be -1 or nonnegative")
	}
	const maxSeed = int64(1<<63 - 1)
	if seed >= 0 && int64(count-1) > maxSeed-seed {
		return 0, errors.New("image batch seed would exceed the supported integer range")
	}
	return seed, nil
}

func imageGenerationFor(request imageJobRequest, profile imageProfile) (*imageGenerationSummary, error) {
	count, err := imageBatchCount(request)
	if err != nil {
		return nil, err
	}
	if _, err = imageBatchSeed(request, profile, count); err != nil {
		return nil, err
	}
	profile = cloneImageProfile(profile)
	refs := append([]string{}, request.ReferenceImages...)
	if request.InputAssetID != "" {
		refs = append(refs, request.InputAssetID)
	}
	if len(refs) > 4 {
		return nil, errors.New("at most four reference images are supported per job")
	}
	preferReference := profile.PreserveReferenceSize && len(refs) > 0
	if request.Width != 0 && !preferReference {
		profile.Width = request.Width
	}
	if request.Height != 0 && !preferReference {
		profile.Height = request.Height
	}
	if request.Steps != 0 {
		profile.Steps = request.Steps
	}
	if request.CFGScale != nil {
		profile.CFGScale = *request.CFGScale
	}
	if request.Sampler != "" {
		profile.Sampler = request.Sampler
	}
	if err := imagegen.ValidateProfile(profile.Profile); err != nil {
		return nil, err
	}
	return &imageGenerationSummary{UserPrompt: request.Prompt, UserNegativePrompt: request.NegativePrompt,
		Prompt: strings.TrimSpace(strings.TrimSpace(profile.PromptPrefix) + "\n" + request.Prompt), NegativePrompt: strings.TrimSpace(strings.TrimSpace(profile.NegativePrompt) + "\n" + request.NegativePrompt),
		ProfileID: profile.ID, Profile: profile, ReferenceAssetIDs: refs, Count: count}, nil
}

// Resolve the registered original before admission/unloading and again before
// execution. The stored generation profile records the actual native canvas,
// while the saved profile and idempotent request retain their original values.
func (v *imageService) resolveImageGenerationDimensions(generation *imageGenerationSummary) error {
	if !generation.Profile.PreserveReferenceSize || len(generation.ReferenceAssetIDs) == 0 {
		return nil
	}
	v.mu.Lock()
	asset, ok := v.assets[generation.ReferenceAssetIDs[0]]
	v.mu.Unlock()
	if !ok {
		return errors.New("reference image is not a registered image asset")
	}
	file, err := openImageAsset(asset)
	if err != nil {
		return err
	}
	file.Close()
	profile, err := imagegen.ResolveProfileDimensions(generation.Profile.Profile, imagegen.Request{
		ReferenceImages: []string{filepath.Join(asset.OutputRoot, asset.relativePath())},
	})
	if err != nil {
		return err
	}
	generation.Profile.Profile = profile
	return nil
}

func imageBatchFilename(index int) string {
	if index == 1 {
		return "image.png"
	}
	return fmt.Sprintf("image-%d.png", index)
}

func (v *imageService) setImageBatchProgress(job *imageJob, index, count int) {
	v.mu.Lock()
	defer v.mu.Unlock()
	job.Progress.BatchIndex = index
	job.Progress.BatchCount = count
}

func (v *imageService) registerImageResult(job *imageJob, result imagegen.Result, filename string, index, count int) error {
	expected := filepath.Join(job.Config.OutputDir, job.ID, filename)
	if filepath.Clean(result.Path) != filepath.Clean(expected) {
		return errors.New("image runner returned an unexpected output path")
	}
	asset := imageAssetRecord{imageAsset: imageAsset{AssetID: job.ID + "/" + filename, RequestID: job.RequestID, SessionID: job.SessionID, JobID: job.ID, Filename: filename, MIMEType: "image/png", Path: expected,
		Width: result.Width, Height: result.Height, Seed: result.Seed, DurationSeconds: result.DurationSeconds, BatchIndex: index, BatchCount: count, ParentAssetID: job.ParentAssetID}, OutputRoot: job.Config.OutputDir}
	if job.Generation != nil {
		asset.ReferenceAssetIDs = append([]string(nil), job.Generation.ReferenceAssetIDs...)
	}
	file, err := openImageAsset(asset)
	if err != nil {
		return err
	}
	file.Close()
	v.mu.Lock()
	defer v.mu.Unlock()
	v.assets[asset.AssetID] = asset
	if err = writeImageJSON(filepath.Join(v.root, "assets.json"), v.assets); err != nil {
		delete(v.assets, asset.AssetID)
		return err
	}
	job.Results = append(job.Results, asset.imageAsset)
	if job.Result == nil {
		first := asset.imageAsset
		job.Result = &first
	}
	// Each published output is recoverable after a restart, even if a later
	// batch image fails. A history error never rolls back the registered asset.
	if err = v.persistJobsLocked(); err != nil {
		return fmt.Errorf("image saved but job history could not be persisted: %w", err)
	}
	return nil
}
