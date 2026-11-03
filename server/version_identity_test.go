package server

import (
	"encoding/json"
	"net/http/httptest"
	"net/url"
	"sync"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
	"github.com/stretchr/testify/require"
)

func versionProbe(t *testing.T, server *Server, query string, header ...string) map[string]any {
	t.Helper()
	response := httptest.NewRecorder()
	context, _ := gin.CreateTestContext(response)
	context.Request = httptest.NewRequest("GET", "/api/version"+query, nil)
	if len(header) > 0 {
		context.Request.Header.Set("X-Wildllama-Profile-Model", header[0])
	}
	server.versionHandler(context)
	require.Equal(t, 200, response.Code)
	var body map[string]any
	require.NoError(t, json.Unmarshal(response.Body.Bytes(), &body))
	return body["wildllama"].(map[string]any)
}

func TestVersionProfileIdentityStableAndDistinct(t *testing.T) {
	first, second := &Server{}, &Server{}
	identity := versionProbe(t, first, "")["instance_id"]
	require.NotEmpty(t, identity)
	require.Equal(t, identity, versionProbe(t, first, "")["instance_id"])
	require.NotEqual(t, identity, versionProbe(t, second, "")["instance_id"])
	require.Equal(t, true, versionProbe(t, first, "")["features"].(map[string]any)["profile_identity"])
	var pending sync.WaitGroup
	for range 8 {
		pending.Go(func() { require.Equal(t, identity, versionProbe(t, first, "")["instance_id"]) })
	}
	pending.Wait()
}

func TestVersionProfileProbeTracksManifestChangesWithoutRunner(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	name := model.ParseName("version-profile:latest")
	write := func(data string) string {
		require.NoError(t, manifest.WriteManifestData(name, []byte(data)))
		mf, err := manifest.ParseNamedManifest(name)
		require.NoError(t, err)
		return mf.Digest()
	}
	first := write(`{"schemaVersion":2,"layers":[]}`)
	server := &Server{} // No scheduler and no runnable weights.
	query := "?model=" + url.QueryEscape("version-profile:latest")
	probe := versionProbe(t, server, query)["profile_check"].(map[string]any)
	require.Equal(t, first, probe["digest"])
	headerProbe := versionProbe(t, server, "", "version-profile:latest")["profile_check"].(map[string]any)
	require.Equal(t, probe, headerProbe)
	second := write(`{"schemaVersion":2,"layers":[],"updated":true}`)
	require.NotEqual(t, first, second)
	probe = versionProbe(t, server, query)["profile_check"].(map[string]any)
	require.Equal(t, second, probe["digest"])
	require.Nil(t, server.sched)
}

func TestVersionProfileProbeInvalidAndMissingPreserveVersionResponse(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	for _, query := range []string{"?model=../escape", "?model=missing:latest", "?model="} {
		probe := versionProbe(t, &Server{}, query)
		require.NotEmpty(t, probe["instance_id"])
		require.Empty(t, probe["profile_check"].(map[string]any)["digest"])
		require.NotEmpty(t, probe["profile_check"].(map[string]any)["error"])
	}
}

func TestVersionProfileHeaderProbeReturnsFreshNamedDigest(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	name := model.ParseName("header-profile:latest")
	server := &Server{}
	probe := func() map[string]any {
		response := httptest.NewRecorder()
		context, _ := gin.CreateTestContext(response)
		context.Request = httptest.NewRequest("GET", "/api/version?model=other-profile:latest", nil)
		context.Request.Header.Set("X-Wildllama-Profile-Model", "header-profile:latest")
		server.versionHandler(context)
		require.Equal(t, 200, response.Code)
		var value struct {
			Wildllama struct {
				Check map[string]any `json:"profile_check"`
			} `json:"wildllama"`
		}
		require.NoError(t, json.Unmarshal(response.Body.Bytes(), &value))
		return value.Wildllama.Check
	}
	require.NoError(t, manifest.WriteManifestData(name, []byte(`{"schemaVersion":2,"layers":[]}`)))
	first := probe()
	require.Equal(t, "header-profile:latest", first["model"])
	require.NotEmpty(t, first["digest"])
	require.NoError(t, manifest.WriteManifestData(name, []byte(`{"schemaVersion":2,"layers":[],"changed":true}`)))
	second := probe()
	require.NotEqual(t, first["digest"], second["digest"])
	require.Empty(t, second["error"])
	require.Nil(t, server.sched)
}
