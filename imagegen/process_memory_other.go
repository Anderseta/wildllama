//go:build !windows && !linux

package imagegen

func newImageWorkingSet(_ int) (peak func() uint64, close func()) {
	return nil, nil
}
