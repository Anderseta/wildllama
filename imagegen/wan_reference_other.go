//go:build !linux && !windows

package imagegen

func newWanReference(size int64) wanReference { return wanReference{} }
