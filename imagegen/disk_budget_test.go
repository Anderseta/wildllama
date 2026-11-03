package imagegen

import (
	"context"
	"errors"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestImageDiskBudgetAccountsRemainingBytesAndReserve(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "not-yet-created", "output.png")
	probe := func(actual string) (uint64, error) { require.Equal(t, root, actual); return 200, nil }
	require.NoError(t, checkDiskSpace(path, 100, 100, probe))
	file := filepath.Join(root, "partial.part")
	require.NoError(t, os.WriteFile(file, []byte("partial"), 0600))
	require.NoError(t, checkDiskSpace(file, 100, 100, probe))
	require.ErrorContains(t, checkDiskSpace(path, 101, 100, probe), "insufficient free disk space")
	require.Error(t, checkDiskSpace(path, math.MaxInt64, 1, probe))
	require.Error(t, checkDiskSpace(path, 1, 1, func(string) (uint64, error) { return 0, errors.New("unavailable") }))
	require.NoError(t, CheckDiskSpace(WithDiskReserve(context.Background(), 0), "missing", math.MaxInt64))
}
