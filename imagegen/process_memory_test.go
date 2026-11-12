//go:build windows || linux

package imagegen

import (
	"os"
	"testing"
)

func TestImageWorkingSetReadsProcessPeak(t *testing.T) {
	peak, close := newImageWorkingSet(os.Getpid())
	if close == nil {
		t.Skip("process memory accounting is unavailable")
	}
	defer close()
	if bytes := peak(); bytes == 0 {
		t.Fatal("process memory accounting did not report a working set")
	}
}
