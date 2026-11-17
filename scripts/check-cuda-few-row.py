"""Small numerical/performance matrix for the opt-in CUDA few-row kernel.

Requires a built test-backend-ops and explicitly selected temporary directory.
Uses synthetic tensors, not model weights. Results are not model tokens/s.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bin-directory', type=Path, required=True)
    parser.add_argument('--work-directory', type=Path, required=True)
    parser.add_argument('--mode', choices=('test', 'perf', 'all'), default='all')
    args = parser.parse_args()
    binary = args.bin_directory.resolve()
    work = args.work_directory.resolve()
    work.mkdir(parents=True, exist_ok=True)
    if os.name == 'nt':
        dll_directory = os.add_dll_directory(str(binary))
        base = ctypes.CDLL(str(binary / 'ggml-base.dll'))
        executable = binary / 'test-backend-ops.exe'
    else:
        base = ctypes.CDLL(str(binary / 'libggml-base.so'))
        executable = binary / 'test-backend-ops'
    base.ggml_type_size.argtypes = [ctypes.c_int]
    base.ggml_type_size.restype = ctypes.c_size_t
    base.ggml_blck_size.argtypes = [ctypes.c_int]
    base.ggml_blck_size.restype = ctypes.c_int64
    base.ggml_op_name.argtypes = [ctypes.c_int]
    base.ggml_op_name.restype = ctypes.c_char_p
    # Public GGML enum values of the pinned backend; fail if they ever drift.
    base.ggml_type_name.argtypes = [ctypes.c_int]
    base.ggml_type_name.restype = ctypes.c_char_p
    types = {'q8_0': 8, 'q3_K': 11, 'q4_K': 12, 'q5_K': 13,
             'q6_K': 14, 'iq4_nl': 20, 'iq3_s': 21, 'iq4_xs': 23}
    mul_mat = next(i for i in range(0, 40) if base.ggml_op_name(i) == b'MUL_MAT')
    lines = []
    for name, kind in types.items():
        assert base.ggml_type_name(kind).decode().lower() == name.lower()
        block, size = base.ggml_blck_size(kind), base.ggml_type_size(kind)
        for n in (2, 3, 5, 8, 17, 24):
            for m, k in ((4096, 4096), (4097, 5120)):
                row = size * k // block
                source_a = [kind, k, m, 1, 1, size, row, row*m, row*m]
                source_b = [0, k, n, 1, 1, 4, k*4, k*n*4, k*n*4]
                values = [mul_mat, 0, m, n, 1, 1, 0, 2, *source_a, *source_b]
                lines.append(' '.join(map(str, values)) + f' {name}-m{m}-k{k}-n{n}')
    fixture = work / 'few-row-ops.txt'
    fixture.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    results = []
    for enabled in ('0', '1'):
        for mode in (('test', 'perf') if args.mode == 'all' else (args.mode,)):
            env = dict(os.environ, WILDLLAMA_CUDA_FEW_ROW=enabled)
            command = [str(executable), mode, '-b', 'CUDA0', '--test-file', str(fixture)]
            with (work / f'{mode}-{enabled}.log').open('w', encoding='utf-8') as output:
                result = subprocess.run(command, cwd=binary, env=env, stdout=output,
                                        stderr=subprocess.STDOUT, timeout=600)
            results.append({'enabled': enabled, 'mode': mode, 'exit': result.returncode})
            print(json.dumps(results[-1]), flush=True)
            if result.returncode:
                (work / f'{args.mode}-results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
                return result.returncode
    (work / f'{args.mode}-results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
