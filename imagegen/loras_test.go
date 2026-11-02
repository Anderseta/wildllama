package imagegen

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoRALibraryInstallVerifiedAndNoProfileMutation(t *testing.T) {
	dir := t.TempDir()
	source := filepath.Join(dir, "source.safetensors")
	writeSafeFixture(t, source, []fixtureTensor{
		{"transformer_blocks.0.attn.to_q.lora_A.default.weight", "BF16", []int64{2, 4}, make([]byte, 16)},
		{"transformer_blocks.0.attn.to_q.lora_B.default.weight", "BF16", []int64{4, 2}, make([]byte, 16)},
	})
	data, err := os.ReadFile(source)
	if err != nil {
		t.Fatal(err)
	}
	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { requests++; w.Write(data) }))
	defer server.Close()
	c := Component{Repo: "fixture/lora", Revision: strings.Repeat("a", 40), Filename: "nested/adapter.safetensors", Size: int64(len(data)), SHA256: digest(data)}
	models := filepath.Join(dir, "models")
	asset, err := installLoRA(context.Background(), models, c, nil, server.Client(), server.URL)
	if err != nil {
		t.Fatal(err)
	}
	if !asset.Ready || asset.ID != c.SHA256 || !strings.Contains(asset.Path, filepath.Join("loras", c.SHA256)) {
		t.Fatal(asset)
	}
	library, err := LoRALibrary(models)
	if err != nil || len(library) != 1 || !library[0].Ready {
		t.Fatal(library, err)
	}
	if _, err := installLoRA(context.Background(), models, c, nil, server.Client(), server.URL); err != nil {
		t.Fatal(err)
	}
	if requests != 1 {
		t.Fatal("verified LoRA downloaded again")
	}
	if _, err := os.Stat(filepath.Join(models, "config.json")); !os.IsNotExist(err) {
		t.Fatal("install changed a generation profile")
	}
	actual, _ := os.ReadFile(source)
	if string(actual) != string(data) {
		t.Fatal("source fixture changed")
	}
}

func TestNonLoRAIsNotPublished(t *testing.T) {
	dir := t.TempDir()
	source := filepath.Join(dir, "not-lora.safetensors")
	writeSafeFixture(t, source, []fixtureTensor{{"ordinary.weight", "BF16", []int64{2, 4}, make([]byte, 16)}})
	data, _ := os.ReadFile(source)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.Write(data) }))
	defer server.Close()
	c := Component{Repo: "fixture/model", Revision: strings.Repeat("b", 40), Filename: "not-lora.safetensors", Size: int64(len(data)), SHA256: digest(data)}
	models := filepath.Join(dir, "models")
	if _, err := installLoRA(context.Background(), models, c, nil, server.Client(), server.URL); err == nil {
		t.Fatal("ordinary model accepted as LoRA")
	}
	library, err := LoRALibrary(models)
	if err != nil || len(library) != 0 {
		t.Fatal("invalid LoRA published", library, err)
	}
}

type testRoundTrip func(*http.Request) (*http.Response, error)

func (f testRoundTrip) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }
func TestDownloadTokenDoesNotCrossRedirectHost(t *testing.T) {
	calls := 0
	transport := credentialTransport{token: "ephemeral-test-token", next: testRoundTrip(func(r *http.Request) (*http.Response, error) {
		calls++
		if r.URL.Hostname() == "huggingface.co" {
			if r.Header.Get("Authorization") != "Bearer ephemeral-test-token" {
				t.Fatal("token missing on origin")
			}
			return &http.Response{StatusCode: 302, Header: http.Header{"Location": []string{"https://unrelated.example/file"}}, Body: io.NopCloser(strings.NewReader("")), Request: r}, nil
		}
		if r.Header.Get("Authorization") != "" {
			t.Fatal("credential crossed redirect host")
		}
		return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader("ok")), Request: r}, nil
	})}
	client := http.Client{Transport: transport}
	request, _ := http.NewRequestWithContext(WithDownloadToken(context.Background(), "ephemeral-test-token"), http.MethodGet, "https://huggingface.co/fixture/repo", nil)
	request.Header.Set("Authorization", "Bearer original-test-token")
	response, err := client.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	response.Body.Close()
	if calls != 2 {
		t.Fatal(calls)
	}
}
