//go:build windows

package imagegen

import (
	"io"
	"os/exec"
	"sync"
	"syscall"
	"unsafe"

	"golang.org/x/sys/windows"
)

func prepareCommand(cmd *exec.Cmd) { cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true} }

type childJob struct {
	mu     sync.Mutex
	handle windows.Handle
}

func (j *childJob) Close() error {
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.handle == 0 {
		return nil
	}
	err := windows.CloseHandle(j.handle)
	j.handle = 0
	return err
}
func attachChild(cmd *exec.Cmd) (io.Closer, error) {
	job, err := windows.CreateJobObject(nil, nil)
	if err != nil {
		return nil, err
	}
	limits := windows.JOBOBJECT_EXTENDED_LIMIT_INFORMATION{}
	limits.BasicLimitInformation.LimitFlags = windows.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
	if _, err = windows.SetInformationJobObject(job, windows.JobObjectExtendedLimitInformation, uintptr(unsafe.Pointer(&limits)), uint32(unsafe.Sizeof(limits))); err != nil {
		windows.CloseHandle(job)
		return nil, err
	}
	process, err := windows.OpenProcess(windows.PROCESS_SET_QUOTA|windows.PROCESS_TERMINATE, false, uint32(cmd.Process.Pid))
	if err != nil {
		windows.CloseHandle(job)
		return nil, err
	}
	defer windows.CloseHandle(process)
	if err = windows.AssignProcessToJobObject(job, process); err != nil {
		windows.CloseHandle(job)
		return nil, err
	}
	return &childJob{handle: job}, nil
}
