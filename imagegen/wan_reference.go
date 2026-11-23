package imagegen

import (
	"os"
	"os/exec"
	"strconv"
)

// Only successive GPU attempts of the same job can access these exact VAE latents.
type wanReference struct {
	name  string
	file  *os.File
	close func()
	view  func(int, func([]byte) error) error
}

func (m wanReference) pass(cmd *exec.Cmd, key string) {
	name := m.name
	if m.file != nil {
		name = "fd:" + strconv.Itoa(3+len(cmd.ExtraFiles))
		cmd.ExtraFiles = append(cmd.ExtraFiles, m.file)
	}
	if name != "" {
		cmd.Env = append(cmd.Env, key+"="+name)
	}
}
