package imagegen

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
)

func TestMOSSEditAdapterIsAppliedOnlyToReferences(t *testing.T) {
	dir := t.TempDir()
	p := testProfile(t, dir)
	p.ModelID = MOSSRecipeID
	p.Tokenizer = filepath.Join(dir, "tokenizer.json")
	putModel(t, p.Tokenizer, []byte(`{"model":{}}`))
	style := filepath.Join(dir, "style.safetensors")
	edit := filepath.Join(dir, "edit.safetensors")
	writeSafeFixture(t, style, []fixtureTensor{
		{"layers.0.proj.k.lora_A.weight", "BF16", []int64{2, 4}, make([]byte, 16)},
		{"layers.0.proj.k.lora_B.weight", "BF16", []int64{4, 2}, make([]byte, 16)},
	})
	writeSafeFixture(t, edit, []fixtureTensor{
		{"layers.0.proj.k.lora_A", "BF16", []int64{2, 4}, make([]byte, 16)},
		{"layers.0.proj.k.lora_B", "BF16", []int64{4, 2}, make([]byte, 16)},
	}, map[string]string{"adapter_config": `{"format":"moss-edit-lora","format_version":1,"modules":{"layers.0.proj.k":{"alpha":2,"rank":2}}}`})
	p.LoRAs = []LoRA{{Path: style, Scale: 0.8}}
	reference := filepath.Join(dir, "reference.png")
	referenceSizeHeader(t, reference, 96, 96)
	request := Request{Prompt: "Change the background", ReferenceImages: []string{reference}}
	if err := ValidateRequest(p, request); err == nil {
		t.Fatal("accepted an edit without an edit adapter")
	}
	p.MOSSEditLoRA = &LoRA{Path: edit, Scale: 1}
	if err := ValidateRequest(p, request); err != nil {
		t.Fatal(err)
	}
	if len(p.requestLoRAs(nil)) != 1 || len(p.requestLoRAs(request.ReferenceImages)) != 2 {
		t.Fatal("incorrect style and edit adapter selection")
	}
	args := strings.Join(buildArguments(p, request, dir, "edit.png", 42), "|")
	if !strings.Contains(args, "--ref-image-args|preset=moss_edit,vae_input_max_edge=384|-r|"+reference) || strings.Contains(args, "--llm_vision") {
		t.Fatal("incorrect MOSS edit reference settings", args)
	}
	config := Config{WorkDir: dir, ConditioningCacheMiB: 256}
	key := func(profile Profile, request Request) string {
		t.Helper()
		env, err := conditioningCacheEnvironment(t.Context(), config, profile, request, "fixture")
		if err != nil {
			t.Fatal(err)
		}
		return envValue(env, "WILDIMAGE_CONDITION_NAMESPACE")
	}
	editKey := key(p, request)
	request.ReferenceImages = nil
	textKey := key(p, request)
	p.MOSSEditLoRA = nil
	if editKey == textKey || key(p, request) != textKey {
		t.Fatal("text-only conditioning depends on an unused edit adapter")
	}
}

func TestMOSSInstallBindsTokenizerAndBothLoRAsAcrossResume(t *testing.T) {
	recipe := mossRecipe()
	fixture := testProfile(t, t.TempDir())
	vae, err := os.ReadFile(fixture.VAE)
	if err != nil {
		t.Fatal(err)
	}
	files := map[string][]byte{}
	for i, component := range recipe.Components {
		data := modelBytes()
		switch component.Role {
		case "vae":
			data = vae
		case "tokenizer":
			data = []byte("{\"model\":{}}")
		}
		files[filepath.Base(component.Filename)] = data
		recipe.Components[i].Size = int64(len(data))
		recipe.Components[i].SHA256 = digest(data)
	}
	var resumed atomic.Bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		name := filepath.Base(r.URL.Path)
		data, ok := files[name]
		if !ok {
			http.NotFound(w, r)
			return
		}
		if name == mossFocusedLoRA {
			if r.Header.Get("Range") != "bytes=8-" {
				http.Error(w, "expected resumable range", http.StatusBadRequest)
				return
			}
			resumed.Store(true)
			w.Header().Set("Content-Range", fmt.Sprintf("bytes 8-%d/%d", len(data)-1, len(data)))
			w.WriteHeader(http.StatusPartialContent)
			data = data[8:]
		}
		_, _ = w.Write(data)
	}))
	defer server.Close()

	root := t.TempDir()
	partial := filepath.Join(root, recipe.ID, "lora", mossFocusedLoRA) + ".part"
	if err := os.MkdirAll(filepath.Dir(partial), 0700); err != nil {
		t.Fatal(err)
	}
	putModel(t, partial, files[mossFocusedLoRA][:8])
	bundle, err := installRecipe(context.Background(), root, recipe, false, server.Client(), server.URL, nil)
	if err != nil || !bundle.Ready || !resumed.Load() {
		t.Fatalf("installation: ready=%v resumed=%v err=%v", bundle.Ready, resumed.Load(), err)
	}
	bundles, _, err := Catalog(root)
	if err != nil || len(bundles) != 1 || !bundles[0].Ready {
		t.Fatalf("cannot reload installed MOSS bundle: %v %v", bundles, err)
	}
	p := bundles[0].Profile
	if p.ModelID != MOSSRecipeID || p.PromptPrefix != "moss_xvstyle" || p.Width != 1024 || p.Height != 1024 ||
		p.Steps != 30 || p.CFGScale != 3 || p.Sigmas != "" || p.Scheduler != "discrete" ||
		!p.StageComponents || !p.OffloadToCPU || p.VisionEncoder != "" || len(p.LoRAs) != 1 {
		t.Fatalf("incorrect MOSS defaults: %+v", p)
	}
	if p.LoRAs[0].Path != strings.TrimSuffix(partial, ".part") || p.LoRAs[0].Scale != 0.8 {
		t.Fatalf("incorrect LoRA selection: %+v", p.LoRAs)
	}
	original := filepath.Join(root, recipe.ID, "lora", mossOriginalLoRA)
	if _, err := os.Stat(original); err != nil {
		t.Fatalf("alternative LoRA missing: %v", err)
	}
	args := strings.Join(buildArguments(p, Request{}, root, "out.png", 42), "|")
	if !strings.Contains(args, "--tokenizer|"+p.Tokenizer) || !strings.Contains(args, "--lora-apply-mode|at_runtime") ||
		!strings.Contains(args, "--scheduler|discrete") || strings.Contains(args, "--sigmas") {
		t.Fatalf("incorrect runner arguments: %s", args)
	}
	if err := os.Remove(p.Tokenizer); err != nil {
		t.Fatal(err)
	}
	bundles, _, err = Catalog(root)
	if err != nil || len(bundles) != 1 || bundles[0].Ready || len(bundles[0].Missing) != 1 ||
		!strings.HasPrefix(bundles[0].Missing[0], "tokenizer:") {
		t.Fatalf("missing tokenizer accepted: %v %v", bundles, err)
	}
}
