//go:build !windows

package ml

// Other platforms report physical and swap memory through their native discovery path.
func ReadWindowsMemory() (SystemInfo, error) { return SystemInfo{}, nil }
