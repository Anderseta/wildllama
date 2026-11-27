package imagegen

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestOmniFP8InstallUsesExistingComponentsAndBindsVision(t *testing.T) {
	recipes := Recipes()
	index := slices.IndexFunc(recipes, func(r Recipe) bool { return r.ID == OmniFP8RecipeID })
	if index < 0 {
		t.Fatal("Omni FP8 recipe missing from the catalog")
	}
	recipe := recipes[index]
	dir := t.TempDir()
	fixture := filepath.Join(dir, "fixture.safetensors")
	writeSafeFixture(t, fixture, []fixtureTensor{
		{"weight", "F32", []int64{1}, make([]byte, 4)},
	})
	data, err := os.ReadFile(fixture)
	if err != nil {
		t.Fatal(err)
	}
	for i := range recipe.Components {
		c := &recipe.Components[i]
		c.Size, c.SHA256 = int64(len(data)), digest(data)
		if c.Role != "diffusion_model" {
			path := filepath.Join(dir, recipe.ID, c.Role, filepath.Base(c.Filename))
			if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
				t.Fatal(err)
			}
			putModel(t, path, data)
		}
	}
	downloaded := recipe.Components[0]
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if filepath.Base(r.URL.Path) != downloaded.Filename {
			t.Errorf("unexpected download of %s", r.URL.Path)
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write(data)
	}))
	defer server.Close()
	bundle, err := installRecipe(context.Background(), dir, recipe, false, server.Client(), server.URL, nil)
	if err != nil || !bundle.Ready {
		t.Fatalf("Omni install: ready=%v, err=%v", bundle.Ready, err)
	}
	models, _, err := Catalog(dir)
	if err != nil || len(models) != 1 || !models[0].Ready {
		t.Fatalf("Omni catalog: models=%+v, err=%v", models, err)
	}
	p := models[0].Profile
	if p.ModelID != OmniFP8RecipeID || p.VisionEncoder == "" || p.VisionEncoder != p.TextEncoder {
		t.Fatalf("Omni edit must use the integrated vision encoder: %+v", p)
	}
	if p.Sigmas != "" || p.Steps != 35 || p.CFGScale != 5 || p.VAETiling || !p.StageComponents {
		t.Fatalf("wrong non-distilled Omni profile: %+v", p)
	}
	args := buildArguments(p, Request{ReferenceImages: []string{"reference.png"}}, dir, "out.png", 42)
	if slices.Contains(args, "--llm_vision") || !slices.Contains(args, "reference.png") {
		t.Fatalf("incorrect Omni edit arguments: %v", args)
	}
}
