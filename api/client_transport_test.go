package api

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
)

type transportTestBody struct {
	reader io.Reader
	read   int
	closed bool
}

func (b *transportTestBody) Read(p []byte) (int, error) {
	n, err := b.reader.Read(p)
	b.read += n
	return n, err
}

func (b *transportTestBody) Close() error {
	b.closed = true
	return nil
}

func transportTestClient(status int, body io.ReadCloser) *Client {
	return NewClient(&url.URL{Scheme: "http", Host: "localhost"}, &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: status, Status: fmt.Sprintf("%d %s", status, http.StatusText(status)), Body: body, Header: make(http.Header), Request: r}, nil
	})})
}

func TestClientStreamEmptyHTTPStatus(t *testing.T) {
	for _, status := range []int{http.StatusOK, http.StatusNoContent, http.StatusBadRequest, http.StatusUnauthorized, http.StatusNotFound, http.StatusTooManyRequests, http.StatusInternalServerError} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			body := &transportTestBody{reader: strings.NewReader("")}
			called := false
			err := transportTestClient(status, body).stream(t.Context(), http.MethodPost, "/api/chat", nil, func([]byte) error { called = true; return nil })
			if status < http.StatusBadRequest {
				if called || !body.closed || err != nil {
					t.Fatalf("empty success changed: callback=%v closed=%v error=%v", called, body.closed, err)
				}
				return
			}
			if called || !body.closed || err == nil {
				t.Fatalf("empty failure disappeared: callback=%v closed=%v error=%v", called, body.closed, err)
			}
			if status == http.StatusUnauthorized {
				var auth AuthorizationError
				if !errors.As(err, &auth) || auth.StatusCode != status || auth.Status != "401 Unauthorized" {
					t.Fatalf("authorization metadata lost: %T %v", err, err)
				}
			} else {
				var failure StatusError
				if !errors.As(err, &failure) || failure.StatusCode != status || failure.Status == "" {
					t.Fatalf("HTTP metadata lost: %T %v", err, err)
				}
			}
		})
	}
}

func TestClientStreamPreservesLargeFramesAndLimit(t *testing.T) {
	content := strings.Repeat("ação", 128*1024)
	payload := `{"done":true,"response":"` + content + `"}` + "\n"
	body := &transportTestBody{reader: strings.NewReader(payload)}
	var got GenerateResponse
	err := transportTestClient(http.StatusOK, body).Generate(t.Context(), &GenerateRequest{}, func(r GenerateResponse) error { got = r; return nil })
	if err != nil || got.Response != content || !body.closed {
		t.Fatalf("large frame was changed: length=%d closed=%v error=%v", len(got.Response), body.closed, err)
	}
	tooLarge := &transportTestBody{reader: strings.NewReader(`{"response":"` + strings.Repeat("x", maxBufferSize) + `"}` + "\n")}
	err = transportTestClient(http.StatusOK, tooLarge).stream(t.Context(), http.MethodPost, "/api/chat", nil, func([]byte) error { t.Error("oversized frame reached callback"); return nil })
	if err == nil || !strings.Contains(err.Error(), "token too long") || !tooLarge.closed {
		t.Fatalf("existing frame bound changed: closed=%v error=%v", tooLarge.closed, err)
	}
}

func TestClientSuccessfulNoResultDrainsBody(t *testing.T) {
	for _, tc := range []struct {
		name string
		body io.Reader
		want int
		err  error
	}{
		{"large success", strings.NewReader(strings.Repeat("discard", 1<<17)), 7 << 17, nil},
		{"read failure", &failingReader{data: []byte("partial"), err: io.ErrUnexpectedEOF}, len("partial"), io.ErrUnexpectedEOF},
	} {
		t.Run(tc.name, func(t *testing.T) {
			body := &transportTestBody{reader: tc.body}
			err := transportTestClient(http.StatusOK, body).do(t.Context(), http.MethodDelete, "/api/delete", nil, nil)
			if !errors.Is(err, tc.err) || body.read != tc.want || !body.closed {
				t.Fatalf("drain=%d closed=%v error=%v", body.read, body.closed, err)
			}
		})
	}
	// Ignoring a successful body must not hide an actual HTTP failure.
	body := &transportTestBody{reader: strings.NewReader(`{"error":"still in use"}`)}
	err := transportTestClient(http.StatusConflict, body).do(t.Context(), http.MethodDelete, "/api/delete", nil, nil)
	var failure StatusError
	if !errors.As(err, &failure) || failure.StatusCode != http.StatusConflict || failure.ErrorMessage != "still in use" || !body.closed {
		t.Fatalf("failure suppressed while draining: %T %v", err, err)
	}
}

func TestClientStreamCancellationAndCallbackFailureClose(t *testing.T) {
	t.Run("callback", func(t *testing.T) {
		body := &transportTestBody{reader: strings.NewReader("{}\n{}\n")}
		stop := errors.New("stop consuming")
		err := transportTestClient(http.StatusOK, body).stream(t.Context(), http.MethodPost, "/api/chat", nil, func([]byte) error { return stop })
		if !errors.Is(err, stop) || !body.closed {
			t.Fatalf("callback cancellation=%v closed=%v", err, body.closed)
		}
	})
	t.Run("request context", func(t *testing.T) {
		cancelled := make(chan struct{})
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			fmt.Fprint(w, "{}\n")
			w.(http.Flusher).Flush()
			<-r.Context().Done()
			close(cancelled)
		}))
		defer server.Close()
		base, _ := url.Parse(server.URL)
		ctx, cancel := context.WithCancel(t.Context())
		defer cancel()
		err := NewClient(base, server.Client()).stream(ctx, http.MethodPost, "/api/chat", nil, func([]byte) error { cancel(); return nil })
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("context cancellation lost: %v", err)
		}
		<-cancelled
	})
}

// Keep network transfer out of these component measurements. Wrapping the
// string reader deliberately hides WriterTo, as a real HTTP body does.
func benchmarkClientTransport(b *testing.B, payload string, stream bool) {
	b.Setenv("OLLAMA_AUTH", "")
	b.ReportAllocs()
	b.SetBytes(int64(len(payload)))
	client := NewClient(&url.URL{Scheme: "http", Host: "localhost"}, &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusOK, Body: &transportTestBody{reader: strings.NewReader(payload)}, Header: make(http.Header), Request: r}, nil
	})})
	for b.Loop() {
		var err error
		if stream {
			err = client.stream(b.Context(), http.MethodPost, "/api/chat", nil, func([]byte) error { return nil })
		} else {
			err = client.do(b.Context(), http.MethodDelete, "/api/delete", nil, nil)
		}
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkClientTransport(b *testing.B) {
	for _, size := range []int{64, 1 << 20, 7 << 20} {
		b.Run(fmt.Sprintf("stream-%d", size), func(b *testing.B) {
			benchmarkClientTransport(b, `{"response":"`+strings.Repeat("x", size)+`"}`+"\n", true)
		})
	}
	b.Run("discard-8MiB", func(b *testing.B) { benchmarkClientTransport(b, strings.Repeat("x", 8<<20), false) })
}

func TestOperationStreamRequiresTerminalFrame(t *testing.T) {
	callbackErr := errors.New("callback stopped")
	for _, tc := range []struct {
		name, partial, terminal string
		run                     func(*Client, func() error) error
	}{
		{"chat", `{"message":{"content":"partial"}}`, `{"done":true}`, func(c *Client, visit func() error) error {
			return c.Chat(t.Context(), &ChatRequest{}, func(ChatResponse) error { return visit() })
		}},
		{"generate", `{"response":"partial"}`, `{"done":true}`, func(c *Client, visit func() error) error {
			return c.Generate(t.Context(), &GenerateRequest{}, func(GenerateResponse) error { return visit() })
		}},
		{"pull", `{"status":"pulling manifest"}`, `{"status":"success"}`, func(c *Client, visit func() error) error {
			return c.Pull(t.Context(), &PullRequest{}, func(ProgressResponse) error { return visit() })
		}},
		{"push", `{"status":"uploading"}`, `{"status":"success"}`, func(c *Client, visit func() error) error {
			return c.Push(t.Context(), &PushRequest{}, func(ProgressResponse) error { return visit() })
		}},
		{"create", `{"status":"creating"}`, `{"status":"success"}`, func(c *Client, visit func() error) error {
			return c.Create(t.Context(), &CreateRequest{}, func(ProgressResponse) error { return visit() })
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, test := range []struct {
				input                string
				calls                int
				callbackErr, wantErr error
			}{
				{"", 0, nil, io.ErrUnexpectedEOF},
				{tc.partial, 1, nil, io.ErrUnexpectedEOF},
				{tc.partial + "\n" + tc.terminal + "\n", 2, nil, nil},
				{tc.terminal + "\n", 1, callbackErr, callbackErr},
			} {
				reader := io.Reader(strings.NewReader(test.input))
				if test.wantErr == nil {
					reader = io.MultiReader(reader, &failingReader{err: errors.New("read past terminal frame")})
				}
				body := &transportTestBody{reader: reader}
				calls := 0
				err := tc.run(transportTestClient(http.StatusOK, body), func() error { calls++; return test.callbackErr })
				if !errors.Is(err, test.wantErr) || calls != test.calls || !body.closed {
					t.Errorf("input %q: err=%v calls=%d closed=%t", test.input, err, calls, body.closed)
				}
			}
		})
	}
}

func TestOperationStreamDebugAndBufferedCancellation(t *testing.T) {
	for _, generate := range []bool{false, true} {
		body := &transportTestBody{reader: strings.NewReader(`{"_debug_info":{"rendered_template":"hello"}}`)}
		client := transportTestClient(http.StatusOK, body)
		var err error
		if generate {
			err = client.Generate(t.Context(), &GenerateRequest{DebugRenderOnly: true}, func(GenerateResponse) error { return nil })
		} else {
			err = client.Chat(t.Context(), &ChatRequest{DebugRenderOnly: true}, func(ChatResponse) error { return nil })
		}
		if err != nil || !body.closed {
			t.Fatalf("debug generate=%v: %v", generate, err)
		}
	}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	body := &transportTestBody{reader: strings.NewReader("{}\n{}\n{\"done\":true}\n")}
	calls := 0
	err := transportTestClient(http.StatusOK, body).Chat(ctx, &ChatRequest{}, func(ChatResponse) error {
		calls++
		cancel()
		return nil
	})
	if !errors.Is(err, context.Canceled) || calls != 1 {
		t.Fatalf("cancellation: %v, calls=%d", err, calls)
	}
}
