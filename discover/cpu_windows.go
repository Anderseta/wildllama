package discover

import "github.com/ollama/ollama/ml"

type MEMORYSTATUSEX = ml.MEMORYSTATUSEX

func GetCPUMem() (memInfo, error) {
	s, err := ml.ReadWindowsMemory()
	return memInfo{
		TotalMemory: s.TotalMemory, FreeMemory: s.FreeMemory,
		CommitLimit: s.CommitLimit, CommitAvailable: s.CommitAvailable,
	}, err
}

func IsNUMA() bool {
	// numa support in ggml is linux only
	return false
}
