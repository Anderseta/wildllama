package middleware

import (
	"encoding/json"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
)

// Account for the whole adapter, including the native JSON round trip when
// a protocol writer has no direct frame path.
func BenchmarkNativeMessageFrames(b *testing.B) {
	gin.SetMode(gin.TestMode)
	for _, p := range generationProgressProtocols()[1:] {
		b.Run(p.name, func(b *testing.B) {
			recorder := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(recorder)
			frame := api.ChatResponse{Model: "test", Message: api.Message{Role: "assistant", Content: " pedaço"}}
			last := api.ChatResponse{Model: "test", Done: true, DoneReason: "stop"}
			b.ReportAllocs()
			for b.Loop() {
				recorder.Body.Reset()
				w := p.writer(c.Writer, true)
				direct, _ := w.(interface{ WriteModelResponse(api.ChatResponse) error })
				enc := json.NewEncoder(w)
				for i := 0; i <= 128; i++ {
					chunk := frame
					if i == 128 {
						chunk = last
					}
					var err error
					if direct != nil {
						err = direct.WriteModelResponse(chunk)
					} else {
						err = enc.Encode(chunk)
					}
					if err != nil {
						b.Fatal(err)
					}
				}
			}
		})
	}
}
