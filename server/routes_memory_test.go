package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
)

type placementTestRunner struct{ mockLlm }

func (*placementTestRunner) RuntimeMemory() api.RuntimeMemory {
	return api.RuntimeMemory{Placement: "offload", LayersObserved: true, GPULayers: 54, TotalLayers: 66}
}

func TestPsHandlerPublishesNativePlacementAndHidesIncompleteLoads(t *testing.T) {
	gin.SetMode(gin.TestMode)
	for _, loading := range []bool{false, true} {
		r := withModel(&runnerRef{llama: &placementTestRunner{}, loading: loading}, &Model{ShortName: "fixture:latest"})
		r.publishStatusLocked()
		s := Server{sched: &Scheduler{loaded: map[string]*runnerRef{"fixture": r}}}
		s.sched.publishLoadedSnapshotLocked()
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = httptest.NewRequest(http.MethodGet, "/api/ps", nil)
		s.PsHandler(c)
		var value api.ProcessResponse
		if err := json.Unmarshal(w.Body.Bytes(), &value); err != nil {
			t.Fatal(err)
		}
		if len(value.Models) != 1 {
			t.Fatalf("unexpected response: %s", w.Body.String())
		}
		m := value.Models[0].Memory
		if loading {
			if m != nil {
				t.Fatal("unconfirmed loading placement was published")
			}
		} else if m == nil || m.GPULayers != 54 || m.TotalLayers != 66 || m.Placement != "offload" {
			t.Fatalf("placement did not reach API: %+v", m)
		}
	}
}
