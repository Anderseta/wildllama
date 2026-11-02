//go:build !windows

package imagegen

import (
	"io"
	"os/exec"
	"sync"
	"syscall"
)

func prepareCommand(cmd *exec.Cmd) { cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true} }

type childGroup struct {
	mu  sync.Mutex
	pid int
}

func (g *childGroup) Close() error {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.pid == 0 {
		return nil
	}
	err := syscall.Kill(-g.pid, syscall.SIGKILL)
	g.pid = 0
	if err == syscall.ESRCH {
		return nil
	}
	return err
}
func attachChild(cmd *exec.Cmd) (io.Closer, error) { return &childGroup{pid: cmd.Process.Pid}, nil }
