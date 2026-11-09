package server

import (
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/imagegen"
	"github.com/stretchr/testify/require"
)

func imageLibraryFixture(t *testing.T) (*imageService, *gin.Engine) {
	t.Helper()
	_, service, router, _, _ := newImageTestService(t)
	service.config.Profiles = []imageProfile{}
	service.config.DefaultProfileID = ""
	require.NoError(t, os.MkdirAll(service.config.ModelsDir, 0700))
	return service, router
}

func imageLibraryAddPipeline(t *testing.T, service *imageService, id string, external string) imagegen.Bundle {
	t.Helper()
	directory := filepath.Join(service.config.ModelsDir, id)
	require.NoError(t, os.MkdirAll(directory, 0700))
	profile := imagegen.DefaultProfile(id)
	profile.ID = "profile-" + id
	bundle := imagegen.Bundle{ID: id, Name: "Pipeline " + id, Directory: directory, Profile: profile}
	for _, role := range []string{"diffusion_model", "text_encoder", "vae"} {
		path := filepath.Join(directory, role+".gguf")
		if external != "" && role == "text_encoder" {
			path = external
		}
		require.NoError(t, os.WriteFile(path, append([]byte("GGUF"), make([]byte, 28)...), 0600))
		bundle.Components = append(bundle.Components, imagegen.InstalledComponent{
			Component: imagegen.Component{Role: role, Repo: "owner/model", Revision: strings.Repeat("a", 40), Filename: role + ".gguf", Required: true, Size: 32}, Path: path})
		switch role {
		case "diffusion_model":
			bundle.Profile.DiffusionModel = path
		case "text_encoder":
			bundle.Profile.TextEncoder = path
		case "vae":
			bundle.Profile.VAE = path
		}
	}
	require.NoError(t, writeImageJSON(filepath.Join(directory, imagegen.ManifestName), bundle))
	service.config.Profiles = append(service.config.Profiles, imageProfile{Profile: bundle.Profile, Name: bundle.Name})
	service.config.DefaultProfileID = bundle.Profile.ID
	return bundle
}

func imageLibraryRead(t *testing.T, router http.Handler) imageLibrary {
	t.Helper()
	response := imageTestHTTP(router, "GET", "/api/images/library", nil, "")
	require.Equal(t, 200, response.Code, response.Body.String())
	var library imageLibrary
	require.NoError(t, json.Unmarshal(response.Body.Bytes(), &library))
	return library
}

func imageLibraryDelete(t *testing.T, router http.Handler, request map[string]any) int {
	t.Helper()
	data, err := json.Marshal(request)
	require.NoError(t, err)
	return imageTestHTTP(router, "POST", "/api/images/library/delete", data, "application/json").Code
}

func TestImageLibraryListsRolesSourcesAndUnusedComponentsWithoutInference(t *testing.T) {
	service, router := imageLibraryFixture(t)
	bundle := imageLibraryAddPipeline(t, service, "one", "")
	lora := filepath.Join(bundle.Directory, "my-lora.safetensors")
	require.NoError(t, os.WriteFile(lora, []byte("fixture"), 0600))
	service.config.Profiles[0].LoRAs = []imagegen.LoRA{{Path: lora, Scale: 1}}
	orphan := filepath.Join(service.config.ModelsDir, "unused.gguf")
	require.NoError(t, os.WriteFile(orphan, []byte("GGUF"), 0600))
	library := imageLibraryRead(t, router)
	require.Len(t, library.Components, 5)
	require.Len(t, library.Pipelines, 1)
	require.Equal(t, service.config.Revision, library.Revision)
	for _, component := range library.Components {
		if component.Path == orphan {
			require.True(t, component.CanDelete)
		} else {
			require.False(t, component.CanDelete)
			require.Equal(t, []string{bundle.Profile.ID}, component.ProfileIDs)
			if component.Path == lora {
				require.Equal(t, []string{"lora"}, component.Roles)
			} else {
				require.Contains(t, component.SourceURL, "https://huggingface.co/owner/model/blob/")
			}
		}
	}
}

func TestImageLibraryPipelineRequiresProfileConfirmationAndPreservesExternalOriginal(t *testing.T) {
	service, router := imageLibraryFixture(t)
	external := filepath.Join(t.TempDir(), "external-encoder.gguf")
	bundle := imageLibraryAddPipeline(t, service, "imported", external)
	library := imageLibraryRead(t, router)
	request := map[string]any{"kind": "pipeline", "id": bundle.ID, "revision": library.Revision, "delete_files": true}
	require.Equal(t, 409, imageLibraryDelete(t, router, request))
	require.FileExists(t, filepath.Join(bundle.Directory, imagegen.ManifestName))
	request["confirm_profiles"] = true
	require.Equal(t, 200, imageLibraryDelete(t, router, request))
	require.FileExists(t, external)
	require.NoFileExists(t, bundle.Profile.DiffusionModel)
	require.NoFileExists(t, bundle.Profile.VAE)
	require.NoFileExists(t, filepath.Join(bundle.Directory, imagegen.ManifestName))
	require.Empty(t, service.config.Profiles)
	require.Empty(t, service.config.DefaultProfileID)
	require.Equal(t, 409, imageLibraryDelete(t, router, request), "stale revisions cannot delete another model")
}

func TestImageLibraryPipelinePreservesSharedComponentsAndUnrecognizedFiles(t *testing.T) {
	service, router := imageLibraryFixture(t)
	first := imageLibraryAddPipeline(t, service, "first", "")
	second := imageLibraryAddPipeline(t, service, "second", first.Profile.TextEncoder)
	unknown := filepath.Join(first.Directory, "keep-me.txt")
	require.NoError(t, os.WriteFile(unknown, []byte("not a model"), 0600))
	request := map[string]any{"kind": "pipeline", "id": first.ID, "revision": service.config.Revision, "delete_files": true, "confirm_profiles": true}
	require.Equal(t, 200, imageLibraryDelete(t, router, request))
	require.FileExists(t, first.Profile.TextEncoder)
	require.FileExists(t, second.Profile.VAE)
	require.FileExists(t, unknown)
	require.Len(t, service.config.Profiles, 1)
	require.Equal(t, second.Profile.ID, service.config.DefaultProfileID)
	library := imageLibraryRead(t, router)
	for _, component := range library.Components {
		if component.Path == first.Profile.TextEncoder {
			require.Equal(t, []string{second.ID}, component.PipelineIDs)
			require.False(t, component.CanDelete)
		}
	}
}

func TestImageLibraryRemoveOnlyRetainsWeightsForExplicitLaterDeletion(t *testing.T) {
	service, router := imageLibraryFixture(t)
	bundle := imageLibraryAddPipeline(t, service, "one", "")
	require.Equal(t, 200, imageLibraryDelete(t, router, map[string]any{"kind": "pipeline", "id": bundle.ID, "revision": service.config.Revision, "confirm_profiles": true}))
	require.FileExists(t, bundle.Profile.DiffusionModel)
	library := imageLibraryRead(t, router)
	require.Empty(t, library.Pipelines)
	require.Len(t, library.Components, 3)
	for _, component := range library.Components {
		require.True(t, component.CanDelete)
		require.Equal(t, 200, imageLibraryDelete(t, router, map[string]any{"kind": "component", "id": component.ID, "revision": service.config.Revision, "delete_files": true}))
		require.NoFileExists(t, component.Path)
	}
}

func TestImageLibraryBlocksReferencesActiveJobsAndStaleAdmissions(t *testing.T) {
	service, router := imageLibraryFixture(t)
	bundle := imageLibraryAddPipeline(t, service, "one", "")
	component := imageLibraryRead(t, router).Components[0]
	require.Equal(t, 409, imageLibraryDelete(t, router, map[string]any{"kind": "component", "id": component.ID, "revision": service.config.Revision, "delete_files": true}))
	service.jobs = []*imageJob{{ID: "stopping", Status: "cancelled", executing: true}}
	library := imageLibraryRead(t, router)
	require.False(t, library.Pipelines[0].CanRemove)
	require.Equal(t, 409, imageLibraryDelete(t, router, map[string]any{"kind": "pipeline", "id": bundle.ID, "revision": service.config.Revision, "confirm_profiles": true}))
	service.jobs = nil
	oldConfig := cloneImageConfig(service.config)
	service.config.Revision++
	_, err := service.enqueue(&imageJob{Kind: "generate", Config: oldConfig})
	require.ErrorContains(t, err, "configuration changed")
	require.Empty(t, service.jobs)
}

func TestImageLibraryRejectsLinkedRootsAndComponentPaths(t *testing.T) {
	service, router := imageLibraryFixture(t)
	target := filepath.Join(t.TempDir(), "original.gguf")
	require.NoError(t, os.WriteFile(target, []byte("GGUF"), 0600))
	link := filepath.Join(service.config.ModelsDir, "linked.gguf")
	if err := os.Symlink(target, link); err != nil {
		t.Skipf("symbolic links unavailable: %v", err)
	}
	require.Error(t, imageLibrarySafePath(service.config.ModelsDir, link))
	require.Equal(t, 404, imageLibraryDelete(t, router, map[string]any{"kind": "component", "id": imageLibraryID(link), "revision": service.config.Revision, "delete_files": true}))
	require.FileExists(t, target)
	rootLink := filepath.Join(service.root, "linked-models")
	require.NoError(t, os.Symlink(service.config.ModelsDir, rootLink))
	require.Error(t, imageLibrarySafePath(rootLink, filepath.Join(rootLink, "test.gguf")))
}

func TestImageLibraryPreservesComponentsReferencedThroughAliases(t *testing.T) {
	service, router := imageLibraryFixture(t)
	first := imageLibraryAddPipeline(t, service, "first", "")
	link := filepath.Join(service.root, "shared-encoder.gguf")
	if err := os.Symlink(first.Profile.TextEncoder, link); err != nil {
		t.Skipf("symbolic links unavailable: %v", err)
	}
	second := imageLibraryAddPipeline(t, service, "second", link)
	require.Equal(t, 200, imageLibraryDelete(t, router, map[string]any{"kind": "pipeline", "id": first.ID, "revision": service.config.Revision, "delete_files": true, "confirm_profiles": true}))
	require.FileExists(t, first.Profile.TextEncoder)
	library := imageLibraryRead(t, router)
	for _, component := range library.Components {
		if component.ID == imageLibraryID(first.Profile.TextEncoder) {
			require.Equal(t, []string{second.ID}, component.PipelineIDs)
			require.False(t, component.CanDelete)
		}
	}
}

func TestImageLibraryDeletionRecoversBeforeAndAfterConfigPublication(t *testing.T) {
	for _, committed := range []bool{false, true} {
		t.Run(map[bool]string{false: "rollback", true: "finish"}[committed], func(t *testing.T) {
			service, _ := imageLibraryFixture(t)
			original := filepath.Join(service.config.ModelsDir, "unused.gguf")
			temporary := filepath.Join(service.config.ModelsDir, ".delete-fixture")
			require.NoError(t, os.WriteFile(temporary, []byte("GGUF"), 0600))
			receipt := imageLibraryDeletion{ModelsDir: service.config.ModelsDir, Revision: service.config.Revision + 1,
				Moved: []imageLibraryMovedFile{{Original: "unused.gguf", Temporary: ".delete-fixture"}}}
			require.NoError(t, writeImageJSON(filepath.Join(service.root, "library-delete.json"), receipt))
			if committed {
				service.config.Revision++
			}
			require.NoError(t, service.recoverImageLibraryDeletionLocked())
			require.NoFileExists(t, temporary)
			if committed {
				require.NoFileExists(t, original)
			} else {
				require.FileExists(t, original)
			}
			require.NoFileExists(t, filepath.Join(service.root, "library-delete.json"))
		})
	}
}

func TestImageLibraryDeletionRollsBackWhenConfigurationCannotBeSaved(t *testing.T) {
	service, router := imageLibraryFixture(t)
	bundle := imageLibraryAddPipeline(t, service, "one", "")
	revision := service.config.Revision
	// A directory occupying the config filename makes atomic publication fail.
	require.NoError(t, os.Mkdir(filepath.Join(service.root, "config.json"), 0700))
	require.Equal(t, 500, imageLibraryDelete(t, router, map[string]any{"kind": "pipeline", "id": bundle.ID, "revision": revision, "delete_files": true, "confirm_profiles": true}))
	require.Equal(t, revision, service.config.Revision)
	require.Len(t, service.config.Profiles, 1)
	require.FileExists(t, filepath.Join(bundle.Directory, imagegen.ManifestName))
	require.FileExists(t, bundle.Profile.DiffusionModel)
	require.NoFileExists(t, filepath.Join(service.root, "library-delete.json"))
}

func TestImageLibraryInvalidReceiptCannotRemoveOutsideLibrary(t *testing.T) {
	service, _ := imageLibraryFixture(t)
	original := filepath.Join(service.root, "outside.gguf")
	require.NoError(t, os.WriteFile(original, []byte("keep"), 0600))
	receipt := imageLibraryDeletion{ModelsDir: service.config.ModelsDir, Revision: service.config.Revision,
		Moved: []imageLibraryMovedFile{{Original: "../outside.gguf", Temporary: "../.delete-fixture"}}}
	require.NoError(t, writeImageJSON(filepath.Join(service.root, "library-delete.json"), receipt))
	require.Error(t, service.recoverImageLibraryDeletionLocked())
	require.FileExists(t, original)
}

func TestImageLibraryUnreadableManifestDisablesDestructiveDeletion(t *testing.T) {
	service, router := imageLibraryFixture(t)
	directory := filepath.Join(service.config.ModelsDir, "broken")
	require.NoError(t, os.MkdirAll(directory, 0700))
	require.NoError(t, os.WriteFile(filepath.Join(directory, imagegen.ManifestName), []byte("incomplete json"), 0600))
	orphan := filepath.Join(service.config.ModelsDir, "could-be-shared.gguf")
	require.NoError(t, os.WriteFile(orphan, []byte("GGUF"), 0600))
	library := imageLibraryRead(t, router)
	require.True(t, library.Truncated)
	require.False(t, library.Components[0].CanDelete)
	// The invalid registration can still be removed explicitly, retaining files.
	require.Equal(t, 200, imageLibraryDelete(t, router, map[string]any{"kind": "pipeline", "id": "broken", "revision": service.config.Revision}))
	require.FileExists(t, orphan)
}

func TestImageLibraryFindsDeepComponentsAndHandlesMissingRootWithoutWriting(t *testing.T) {
	service, router := imageLibraryFixture(t)
	deep := filepath.Join(service.config.ModelsDir, "a", "b", "c", "d", "e", "vae.gguf")
	require.NoError(t, os.MkdirAll(filepath.Dir(deep), 0700))
	require.NoError(t, os.WriteFile(deep, []byte("GGUF"), 0600))
	library := imageLibraryRead(t, router)
	require.Len(t, library.Components, 1)
	require.Equal(t, deep, library.Components[0].Path)
	require.False(t, library.Truncated)
	service.config.ModelsDir = filepath.Join(service.root, "not-created")
	library = imageLibraryRead(t, router)
	require.Empty(t, library.Components)
	require.Empty(t, library.Pipelines)
	require.Equal(t, 404, imageLibraryDelete(t, router, map[string]any{"kind": "pipeline", "id": "none", "revision": service.config.Revision, "delete_files": true}))
	require.NoDirExists(t, service.config.ModelsDir)
	require.NoFileExists(t, filepath.Join(service.root, "config.json"))
}
