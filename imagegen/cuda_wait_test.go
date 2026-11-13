package imagegen

import (
	"path/filepath"
	"runtime"
	"testing"
)

func TestImageCUDAWaitDefaultAndOverride(t *testing.T) {
	root := t.TempDir()
	config := Config{WorkDir: root, RunnerPath: filepath.Join(root, "sd-cli")}
	for _, value := range []string{"", "auto", "blocking", "spin", "yield"} {
		t.Run(value, func(t *testing.T) {
			t.Setenv("WILDIMAGE_CUDA_WAIT", value)
			want := value
			if want == "" && runtime.GOOS == "windows" {
				want = "blocking"
			}
			if got := envValue(isolatedEnvironment(config, root), "WILDIMAGE_CUDA_WAIT"); got != want {
				t.Fatalf("CUDA wait policy %q, want %q", got, want)
			}
		})
	}
}
