"""Read-only native bundle and actually loaded module identity for local tests.

Importing this module performs no I/O. Explicit callers can pin a selected native
payload and check that its own child loaded those exact files, using Toolhelp32
on Windows or /proc/PID/maps on Linux. No process is modified or launched here.
"""
import hashlib
import os
from pathlib import Path
import time


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest(executable):
    executable=Path(executable).resolve()
    root=executable.parent
    require(executable.is_file() and root.is_absolute(), "Select an explicit existing native payload")
    require(os.name != "nt" or root.drive.upper() != "C:", "Test payloads must remain outside C:")
    selected=[]
    for path in root.rglob("*"):
        if path.is_file():
            require(path.resolve().is_relative_to(root), "Native bundle file escapes its selected directory")
            selected.append(path)
            require(len(selected) <= 256, "Select the packaged native directory, not a workspace/system directory")
    files={path.relative_to(root).as_posix(): sha(path) for path in sorted(selected)}
    require(files and all(not Path(name).is_absolute() and ".." not in Path(name).parts for name in files), "Invalid native bundle inventory")
    return dict(root=str(root), executable=str(executable), files_sha256=files)


def unchanged(identity):
    require(manifest(identity["executable"]) == identity, "The complete native payload changed during validation")


def loaded_modules(pid):
    """Read the child process's actual loaded files without modifying it."""
    if os.name=="nt":
        import ctypes
        from ctypes import wintypes
        class MODULEENTRY32W(ctypes.Structure):
            _fields_=[("dwSize",wintypes.DWORD),("th32ModuleID",wintypes.DWORD),
                      ("th32ProcessID",wintypes.DWORD),("GlblcntUsage",wintypes.DWORD),
                      ("ProccntUsage",wintypes.DWORD),("modBaseAddr",ctypes.POINTER(wintypes.BYTE)),
                      ("modBaseSize",wintypes.DWORD),("hModule",wintypes.HMODULE),
                      ("szModule",wintypes.WCHAR*256),("szExePath",wintypes.WCHAR*260)]
        kernel=ctypes.WinDLL("kernel32",use_last_error=True)
        kernel.CreateToolhelp32Snapshot.argtypes=[wintypes.DWORD,wintypes.DWORD]
        kernel.CreateToolhelp32Snapshot.restype=wintypes.HANDLE
        kernel.Module32FirstW.argtypes=[wintypes.HANDLE,ctypes.POINTER(MODULEENTRY32W)]
        kernel.Module32FirstW.restype=wintypes.BOOL
        kernel.Module32NextW.argtypes=[wintypes.HANDLE,ctypes.POINTER(MODULEENTRY32W)]
        kernel.Module32NextW.restype=wintypes.BOOL
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        kernel.CloseHandle.restype=wintypes.BOOL
        invalid=ctypes.c_void_p(-1).value
        for attempt in range(10):
            snapshot=kernel.CreateToolhelp32Snapshot(0x8|0x10,pid)
            if snapshot!=invalid:break
            error=ctypes.get_last_error()
            if error!=24 or attempt==9:raise ctypes.WinError(error)
            time.sleep(.02)
        try:
            entry=MODULEENTRY32W()
            entry.dwSize=ctypes.sizeof(entry)
            if not kernel.Module32FirstW(snapshot,ctypes.byref(entry)):
                raise ctypes.WinError(ctypes.get_last_error())
            paths=[]
            while True:
                paths.append(str(entry.szExePath))
                if not kernel.Module32NextW(snapshot,ctypes.byref(entry)):
                    error=ctypes.get_last_error()
                    if error!=18:raise ctypes.WinError(error)
                    break
        finally:kernel.CloseHandle(snapshot)
        method="Toolhelp32 Module32FirstW/Module32NextW"
    else:
        paths=[]
        for line in Path(f"/proc/{pid}/maps").read_text().splitlines():
            fields=line.split(None,5)
            if len(fields)==6 and fields[5].startswith("/"):paths.append(fields[5])
        method="/proc/pid/maps"
    return dict(pid=pid,method=method,modules=[dict(basename=Path(path).name,path=str(Path(path).resolve()))
                for path in sorted(set(paths))])



def verify_loaded_payload(server, identity):
    root=Path(identity["root"])
    actual=loaded_modules(server.process.pid)
    suffix,prefix=(".dll", "") if os.name=="nt" else (".so", "lib")
    expected=[Path(identity["executable"]), server.backend]
    expected += [root/(prefix+name+suffix) for name in
        ("llama-server-impl","llama","llama-common","ggml","ggml-base","ggml-cpu")]
    expected=list(dict.fromkeys(path.resolve() for path in expected))
    required_names={path.name.casefold() for path in expected}
    validate=[]
    for path in expected:
        matches=[row for row in actual["modules"] if row["basename"].casefold()==path.name.casefold()]
        require(len(matches)==1, "Required native module was missing/ambiguous: "+path.name)
        loaded=Path(matches[0]["path"]).resolve()
        require(os.path.normcase(str(loaded))==os.path.normcase(str(path)), "Loaded module came from a different payload: "+str(loaded))
        relative=path.relative_to(root).as_posix()
        digest=sha(loaded)
        require(digest==identity["files_sha256"].get(relative), "Loaded module differs from the frozen bundle: "+relative)
        validate.append(dict(path=str(loaded),sha256=digest))
    actual["validated_required_modules"]=validate
    actual["all_required_module_names"]=sorted(required_names)
    return actual
