package server

import (
	"encoding/json"
	"errors"
	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestInferenceMemoryFailureIsExplicitAndNotRetryable(t *testing.T) {
	got := inferenceErrorPayload(errors.New("out of memory: GPU allocation denied"))
	if got["status"] != 507 || got["code"] != "out_of_memory" || got["retryable"] != false {
		t.Fatalf("ambiguous memory error: %+v", got)
	}
	got = inferenceErrorPayload(errors.New("model compute failed: Compute error."))
	if got["status"] != 422 || got["code"] != "model_compute_failed" || got["retryable"] != false {
		t.Fatalf("native compute failure was treated as a connection: %+v", got)
	}
	if _, ok := inferenceErrorPayload(errors.New("temporary network failure"))["retryable"]; ok {
		t.Fatal("ordinary failures lost their existing retry policy")
	}
}

func TestMemoryFailureSurvivesNonStreamingChatAndModelLoad(t *testing.T) {
	gin.SetMode(gin.TestMode)
	for _, load := range []bool{false, true} {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = httptest.NewRequest(http.MethodPost, "/api/chat", nil)
		failure := errors.New("out of memory: allocation denied")
		if load {
			handleScheduleError(c, "fixture", failure)
		} else {
			stream := false
			ch := make(chan any, 1)
			ch <- inferenceErrorPayload(failure)
			close(ch)
			writeChatResponse(c, api.ChatRequest{Stream: &stream}, ch)
		}
		var body map[string]any
		if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
			t.Fatal(err)
		}
		if w.Code != 507 || body["code"] != "out_of_memory" || body["retryable"] != false {
			t.Fatalf("load=%t: status=%d body=%s", load, w.Code, w.Body.String())
		}
	}
}
