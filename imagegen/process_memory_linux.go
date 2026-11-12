//go:build linux

package imagegen

import (
	"bufio"
	"os"
	"strconv"
	"strings"
)

func newImageWorkingSet(pid int) (peak func() uint64, close func()) {
	if pid <= 0 {
		return nil, nil
	}
	status, err := os.Open("/proc/" + strconv.Itoa(pid) + "/status")
	if err != nil {
		return nil, nil
	}
	return func() uint64 {
		if _, err := status.Seek(0, 0); err != nil {
			return 0
		}
		scanner := bufio.NewScanner(status)
		for scanner.Scan() {
			if line, ok := strings.CutPrefix(scanner.Text(), "VmHWM:"); ok {
				fields := strings.Fields(line)
				if len(fields) == 2 && fields[1] == "kB" {
					kib, _ := strconv.ParseUint(fields[0], 10, 54)
					return kib << 10
				}
				break
			}
		}
		return 0
	}, func() { _ = status.Close() }
}
