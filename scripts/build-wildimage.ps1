#Requires -Version 7.2
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$SourceDirectory,
    [Parameter(Mandatory = $true)][string]$BaseRuntimeDirectory,
    [Parameter(Mandatory = $true)][string]$BuildDirectory,
    [Parameter(Mandatory = $true)][string]$Destination,
    [string]$VisualStudioDirectory = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools',
    [ValidatePattern('^\d+\.\d+\.\d+$')][string]$MsvcVersion = '14.39.33519',
    [string]$CmakePath = 'C:\Python310\Lib\site-packages\cmake\data\bin\cmake.exe',
    [string]$CudaDirectory = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8',
    [string]$CudaArchitectures = '80;89;120',
    [ValidateRange(1, 16)][int]$Parallel = 2,
    [switch]$PrepareOnly
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$repository = Split-Path -Parent $PSScriptRoot
$sourceRevision = 'b9d7176bf88fe465576e0aafbe47ba253ffcfc2c'
$runtimeRevision = 'b9d7176-wildllama-h3-1'
$originalEnvironment = @{}
Get-ChildItem Env: | ForEach-Object { $originalEnvironment[$_.Name] = $_.Value }

function Assert-DataPath([string]$Value) {
    if (-not [IO.Path]::IsPathFullyQualified($Value)) { throw "Use an absolute data path: $Value" }
    $absolute = [IO.Path]::GetFullPath($Value).TrimEnd('\', '/')
    if ($absolute -eq [IO.Path]::GetPathRoot($Value).TrimEnd('\', '/')) { throw 'Choose a dedicated data directory, not a drive root.' }
    if ([IO.Path]::GetPathRoot($absolute) -ieq 'C:\') { throw "Build inputs and outputs must be outside C: $absolute" }
    $ancestor = $absolute
    while ($ancestor) {
        if (Test-Path -LiteralPath $ancestor) {
            $item = Get-Item -LiteralPath $ancestor -Force
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                $resolved = $item.ResolveLinkTarget($true)
                if (-not $resolved -or [IO.Path]::GetPathRoot($resolved.FullName) -ieq 'C:\') {
                    throw "Data path resolves through an unsafe link: $ancestor"
                }
            }
        }
        $parent = Split-Path -Parent $ancestor
        if (-not $parent -or $parent -eq $ancestor) { break }
        $ancestor = $parent
    }
    return $absolute
}

function Assert-Tool([string]$Value) {
    if (-not (Test-Path -LiteralPath $Value -PathType Leaf)) { throw "Required existing tool not found: $Value" }
    return $Value
}

function Invoke-Checked([string]$Executable, [string[]]$Arguments, [string]$Log) {
    $output = @(& $Executable @Arguments 2>&1 | Tee-Object -FilePath $Log)
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        $output | Select-Object -Last 30 | Write-Host
        throw "Command failed ($code): $Executable; log: $Log"
    }
    return $output
}

function Copy-ReplacingFile([string]$Source, [string]$Target) {
    $Target = Assert-DataPath $Target
    if (Test-Path -LiteralPath $Target -PathType Leaf) {
        $existing = Get-Item -LiteralPath $Target -Force
        # Keep timestamps on identical regular files: touching copied headers
        # otherwise invalidates every dependent native compilation unit.
        if (-not ($existing.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            $inputFile = Get-Item -LiteralPath $Source -Force
            if ($existing.Length -eq $inputFile.Length -and
                (Get-FileHash -LiteralPath $Target -Algorithm SHA256).Hash -eq
                (Get-FileHash -LiteralPath $Source -Algorithm SHA256).Hash) {
                return
            }
        }
    }
    $directory = Split-Path -Parent $Target
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
    $temporary = Join-Path $directory ('.' + [IO.Path]::GetFileName($Target) + '.' + [guid]::NewGuid().ToString('N') + '.tmp')
    try {
        Copy-Item -LiteralPath $Source -Destination $temporary
        # Unlink the old directory entry first: an old hard link must never cause
        # replacement bytes to be written into the released runtime's file.
        if (Test-Path -LiteralPath $Target) { Remove-Item -LiteralPath $Target -Force }
        Move-Item -LiteralPath $temporary -Destination $Target
    } finally {
        if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary -Force }
    }
}

function File-Record([string]$Path, [string]$Name) {
    $item = Get-Item -LiteralPath $Path
    return @{ name = $Name; size = $item.Length; sha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
}

try {
    $SourceDirectory = Assert-DataPath $SourceDirectory
    $BaseRuntimeDirectory = Assert-DataPath $BaseRuntimeDirectory
    $BuildDirectory = Assert-DataPath $BuildDirectory
    $Destination = Assert-DataPath $Destination
    foreach ($output in @($Destination, $BuildDirectory)) {
        if ($output -ieq $BaseRuntimeDirectory -or $output.StartsWith($BaseRuntimeDirectory + '\', [StringComparison]::OrdinalIgnoreCase) -or $BaseRuntimeDirectory.StartsWith($output + '\', [StringComparison]::OrdinalIgnoreCase)) {
            throw 'Build and destination directories must not overlap the released base runtime.'
        }
    }
    if ($BuildDirectory -ieq $SourceDirectory -or $BuildDirectory -ieq $Destination -or $Destination -ieq $SourceDirectory) {
        throw 'Use separate source, build and destination directories.'
    }
    $toolDirectory = Join-Path $VisualStudioDirectory "VC/Tools/MSVC/$MsvcVersion/bin/Hostx64/x64"
    $cl = Assert-Tool (Join-Path $toolDirectory 'cl.exe')
    $lib = Assert-Tool (Join-Path $toolDirectory 'lib.exe')
    $dumpbin = Assert-Tool (Join-Path $toolDirectory 'dumpbin.exe')
    $vcvars = Assert-Tool (Join-Path $VisualStudioDirectory 'VC/Auxiliary/Build/vcvars64.bat')
    $ninja = Assert-Tool (Join-Path $VisualStudioDirectory 'Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe')
    $CmakePath = Assert-Tool $CmakePath
    $nvcc = Assert-Tool (Join-Path $CudaDirectory 'bin/nvcc.exe')
    $temporaryDirectory = Join-Path $BuildDirectory 'temp'
    $imports = Join-Path $BuildDirectory 'imports'
    $probeDirectory = Join-Path $BuildDirectory 'abi-probe'
    $cmakeDirectory = Join-Path $BuildDirectory 'cmake'
    New-Item -ItemType Directory -Force -Path $BuildDirectory, $temporaryDirectory, $imports, $probeDirectory | Out-Null
    $env:TEMP = $temporaryDirectory
    $env:TMP = $temporaryDirectory
    $env:VSCMD_SKIP_SENDTELEMETRY = '1'

    $toolsetVersion = (($MsvcVersion -split '\.')[0..1] -join '.')
    $environmentLines = & $env:ComSpec /d /s /c ('call "' + $vcvars + '" -vcvars_ver=' + $toolsetVersion + ' >nul && set')
    if ($LASTEXITCODE -ne 0) { throw 'Could not initialize the existing MSVC 14.39 environment.' }
    foreach ($line in $environmentLines) {
        if ($line -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process') }
    }
    if ($env:VCToolsVersion -ne $MsvcVersion) { throw "Unexpected initialized MSVC version: $env:VCToolsVersion" }
    foreach ($name in @('CL', '_CL_', 'CFLAGS', 'CXXFLAGS', 'CUDAFLAGS', 'LDFLAGS')) { [Environment]::SetEnvironmentVariable($name, $null, 'Process') }
    $env:TEMP = $temporaryDirectory
    $env:TMP = $temporaryDirectory
    $env:CUDA_CACHE_PATH = Join-Path $BuildDirectory 'cuda-cache'
    $env:CUDA_VISIBLE_DEVICES = '-1'
    $env:CUDA_PATH = $CudaDirectory
    $env:CUDA_HOME = $CudaDirectory
    $env:VSCMD_SKIP_SENDTELEMETRY = '1'
    $env:PATH = "$toolDirectory;$(Split-Path -Parent $ninja);$CudaDirectory\bin;$BaseRuntimeDirectory;$env:PATH"
    New-Item -ItemType Directory -Force -Path $env:CUDA_CACHE_PATH | Out-Null

    $actualRevision = (& git -C $SourceDirectory rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $actualRevision -ne $sourceRevision) { throw "Unexpected stable-diffusion.cpp revision: $actualRevision" }
    $submodules = @{}
    foreach ($relative in @('ggml', 'thirdparty/cutlass', 'thirdparty/libwebp', 'thirdparty/libwebm')) {
        $treeEntry = (& git -C $SourceDirectory ls-tree HEAD $relative) -join ''
        if ($LASTEXITCODE -ne 0 -or $treeEntry -notmatch '^160000 commit ([0-9a-f]{40})\s') { throw "Missing pinned submodule: $relative" }
        $expectedRevision = $Matches[1]
        $actual = (& git -C (Join-Path $SourceDirectory $relative) rev-parse HEAD).Trim()
        if ($LASTEXITCODE -ne 0 -or $actual -ne $expectedRevision) { throw "Unexpected submodule revision: $relative" }
        & git -C (Join-Path $SourceDirectory $relative) diff --quiet HEAD
        if ($LASTEXITCODE -ne 0) { throw "Submodule contains modified tracked inputs: $relative" }
        $submodules[$relative] = $actual
    }
    $ggmlRevision = $submodules['ggml']
    if (-not $ggmlRevision.StartsWith('d803c238')) { throw "Unexpected patched GGML revision: $ggmlRevision" }
    $baseManifestPath = Join-Path $BaseRuntimeDirectory 'wildimage-runtime.json'
    $baseManifest = Get-Content -LiteralPath $baseManifestPath -Raw | ConvertFrom-Json
    if ($baseManifest.revision -ne $sourceRevision) { throw 'Base runtime is not the pinned upstream release.' }
    $baseFiles = @{}
    foreach ($entry in $baseManifest.files) {
        if ([IO.Path]::GetFileName($entry.name) -ne $entry.name) { throw 'Base manifest contains a non-local filename.' }
        $record = File-Record (Join-Path $BaseRuntimeDirectory $entry.name) $entry.name
        if ($record.sha256 -ne $entry.sha256 -or $record.size -ne $entry.size) { throw "Base runtime file differs from its manifest: $($entry.name)" }
        $baseFiles[$entry.name] = $record
    }
    $importedFiles = @()
    foreach ($stem in @('ggml', 'ggml-base', 'ggml-cpu-x64', 'ggml-cuda')) {
        $dll = Join-Path $BaseRuntimeDirectory "$stem.dll"
        $record = $baseFiles["$stem.dll"]
        if (-not $record) { throw "Base runtime DLL is not recorded: $stem.dll" }
        $exports = Invoke-Checked $dumpbin @('/nologo', '/exports', $dll) (Join-Path $imports "$stem.exports.txt")
        $symbols = @($exports | ForEach-Object { if ($_ -match '^\s*\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]+\s+(\S+)') { $Matches[1] } })
        if (-not $symbols.Count) { throw "No exports found in $dll" }
        $definition = Join-Path $imports "$stem.def"
        @("LIBRARY $stem.dll", 'EXPORTS') + $symbols | Set-Content -LiteralPath $definition -Encoding ascii
        Invoke-Checked $lib @('/nologo', '/machine:x64', "/def:$definition", "/out:$(Join-Path $imports "$stem.lib")") (Join-Path $imports "$stem.lib.log") | Out-Null
        $importedFiles += $record
    }

    $ggmlInclude = Join-Path $SourceDirectory 'ggml/include'
    $ggmlPrivate = Join-Path $SourceDirectory 'ggml/src'
    @'
#include "ggml.h"
#include <cstdio>
#include <cstring>
#include <string>
int main() {
    const size_t overhead = ggml_tensor_overhead();
    if (GGML_MAX_NAME != 160 || overhead != sizeof(ggml_tensor) + 32) {
        std::fprintf(stderr, "GGML ABI mismatch: header=%zu overhead=%zu name=%d\n", sizeof(ggml_tensor), overhead, GGML_MAX_NAME);
        return 11;
    }
    ggml_init_params params = { overhead + 1024, nullptr, false };
    ggml_context * context = ggml_init(params);
    if (!context) return 12;
    ggml_tensor * tensor = ggml_new_tensor_1d(context, GGML_TYPE_F32, 1);
    const std::string name(130, 'n');
    ggml_set_name(tensor, name.c_str());
    const float value = 42.0f;
    std::memcpy(tensor->data, &value, sizeof(value));
    const bool valid = std::strcmp(ggml_get_name(tensor), name.c_str()) == 0 &&
        std::strcmp(tensor->name, name.c_str()) == 0 && *ggml_get_data_f32(tensor) == value;
    std::printf("GGML ABI: tensor=%zu overhead=%zu name=%zu CPU-value=%.1f %s\n", sizeof(ggml_tensor), overhead, std::strlen(tensor->name), *ggml_get_data_f32(tensor), valid ? "PASS" : "FAIL");
    ggml_free(context);
    return valid ? 0 : 13;
}
'@ | Set-Content -LiteralPath (Join-Path $probeDirectory 'abi-probe.cpp') -Encoding ascii
    Push-Location $probeDirectory
    try {
        Invoke-Checked $cl @('/nologo', '/std:c++17', '/EHsc', '/MD', '/DGGML_SHARED', '/DGGML_BACKEND_SHARED', '/DGGML_MAX_NAME=160', "/I$ggmlInclude", 'abi-probe.cpp', '/Fe:abi-probe.exe', '/Fo:abi-probe.obj', '/Fd:abi-probe.pdb', '/link', (Join-Path $imports 'ggml-base.lib')) (Join-Path $probeDirectory 'compile.log') | Out-Null
        Copy-ReplacingFile (Join-Path $BaseRuntimeDirectory 'ggml-base.dll') (Join-Path $probeDirectory 'ggml-base.dll')
        $probe = Invoke-Checked (Join-Path $probeDirectory 'abi-probe.exe') @() (Join-Path $probeDirectory 'result.log')
        $probe | Write-Host
    } finally { Pop-Location }

    $cmakeIncludes = @($ggmlInclude, $ggmlPrivate) | ForEach-Object { $_.Replace('\', '/') }
    $configuration = @('# Generated from the verified released DLL exports. No GGML compilation.', 'set(ggml_FOUND TRUE)', 'set(GGML_FOUND TRUE)')
    foreach ($pair in @(@('base', 'ggml-base'), @('cpu', 'ggml-cpu-x64'), @('cuda', 'ggml-cuda'), @('ggml', 'ggml'))) {
        $target = $pair[0]
        $stem = $pair[1]
        $configuration += @(
            "add_library(ggml::$target SHARED IMPORTED GLOBAL)",
            "set_target_properties(ggml::$target PROPERTIES",
            "  IMPORTED_LOCATION `"$((Join-Path $BaseRuntimeDirectory "$stem.dll").Replace('\', '/'))`"",
            "  IMPORTED_IMPLIB `"$((Join-Path $imports "$stem.lib").Replace('\', '/'))`"",
            "  INTERFACE_INCLUDE_DIRECTORIES `"$($cmakeIncludes -join ';')`"",
            '  INTERFACE_COMPILE_DEFINITIONS "GGML_SHARED;GGML_BACKEND_SHARED;GGML_MAX_NAME=160")'
        )
    }
    $configuration += 'set_property(TARGET ggml::ggml PROPERTY INTERFACE_LINK_LIBRARIES "ggml::base;ggml::cpu;ggml::cuda")'
    $configuration | Set-Content -LiteralPath (Join-Path $imports 'ggml-config.cmake') -Encoding utf8

    $nativeDirectory = Join-Path $repository 'third_party/wildimage/native'
    $conditioningDirectory = Join-Path $SourceDirectory 'src/conditioning'
    New-Item -ItemType Directory -Force -Path $conditioningDirectory | Out-Null
    $sourceInputs = @()
    foreach ($name in @('wildimage_conditioning_cache.h', 'wildimage_conditioning_store.h', 'wildimage_ram_budget.h')) {
        $header = Join-Path $nativeDirectory $name
        Copy-ReplacingFile $header (Join-Path $conditioningDirectory $name)
        $sourceInputs += File-Record $header "third_party/wildimage/native/$name"
    }
    Copy-ReplacingFile (Join-Path $nativeDirectory 'wildimage_ram_budget.h') (Join-Path $SourceDirectory 'src/core/wildimage_ram_budget.h')
    foreach ($name in @('wildimage_fp8.h', 'wildimage_fp8.cu')) {
        $header = Join-Path $nativeDirectory $name
        $target = if ($name.EndsWith('.cu')) { "src/core/cuda/$name" } else { "src/core/$name" }
        Copy-ReplacingFile $header (Join-Path $SourceDirectory $target)
        $sourceInputs += File-Record $header "third_party/wildimage/native/$name"
    }
    foreach($patchName in @('0001-conditioning-disk-cache.patch','0002-bounded-ram-offload.patch','0003-fp8-shared-packing.patch','0005-int8-workspaces.patch','0006-final-decode-staging.patch','0007-gguf-original-shapes.patch')) {
        $patch = Join-Path $repository "third_party/wildimage/patches/$patchName"
        if(-not (Test-Path -LiteralPath $patch)) {throw "Native patch is absent: $patch"}
        & git -C $SourceDirectory apply --check $patch 2>$null
        if ($LASTEXITCODE -eq 0) {
            Invoke-Checked 'git' @('-C', $SourceDirectory, 'apply', $patch) (Join-Path $BuildDirectory ($patchName+'.log')) | Out-Null
        } else {
            & git -C $SourceDirectory apply --reverse --check $patch 2>$null
            if ($LASTEXITCODE -ne 0) { throw "Native patch cannot be applied or verified as already applied: $patchName" }
        }
        $sourceInputs += File-Record $patch "third_party/wildimage/patches/$patchName"
    }
    $sourceInputs += File-Record $PSCommandPath 'scripts/build-wildimage.ps1'
    $configureArguments = @('-S', $SourceDirectory, '-B', $cmakeDirectory, '-G', 'Ninja',
        "-DCMAKE_MAKE_PROGRAM=$($ninja.Replace('\', '/'))", "-DCMAKE_C_COMPILER=$($cl.Replace('\', '/'))", "-DCMAKE_CXX_COMPILER=$($cl.Replace('\', '/'))",
        "-DCMAKE_CUDA_COMPILER=$($nvcc.Replace('\', '/'))", "-DCMAKE_CUDA_HOST_COMPILER=$($cl.Replace('\', '/'))", '-DCMAKE_BUILD_TYPE=Release',
        '-DCMAKE_TRY_COMPILE_CONFIGURATION=Release', '-DCMAKE_C_FLAGS_DEBUG=/MDd /Ob0 /Od /RTC1 /Z7', '-DCMAKE_CXX_FLAGS_DEBUG=/MDd /Ob0 /Od /RTC1 /Z7',
        "-DCMAKE_INSTALL_PREFIX=$((Join-Path $BuildDirectory 'install').Replace('\', '/'))", "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures",
        '-DSD_BUILD_SHARED_LIBS=ON', '-DSD_BUILD_SHARED_GGML_LIB=OFF', '-DSD_BUILD_EXAMPLES=ON', '-DSD_SERVER_BUILD_FRONTEND=OFF',
        '-DSD_USE_SYSTEM_GGML=ON', '-DSD_USE_UPSTREAM_GGML=OFF', "-Dggml_DIR=$($imports.Replace('\', '/'))",
        '-DSD_CUDA=ON', '-DSD_WEBP=ON', '-DSD_WEBM=ON', '-DSD_USE_SYSTEM_WEBP=OFF', '-DSD_USE_SYSTEM_WEBM=OFF')
    Invoke-Checked $CmakePath $configureArguments (Join-Path $BuildDirectory 'configure.log') | Out-Null
    $preparation = @{ source_revision = $sourceRevision; ggml_revision = $ggmlRevision; submodules = $submodules; msvc_toolset = $MsvcVersion; ggml_max_name = 160; cuda_architectures = $CudaArchitectures; imported_files = $importedFiles; inputs = $sourceInputs }
    $preparation | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $BuildDirectory 'build-inputs.json') -Encoding utf8
    if ($PrepareOnly) {
        Write-Output "WildImage prepared; no core/CUDA build started: $cmakeDirectory"
        return
    }

    Invoke-Checked $CmakePath @('--build', $cmakeDirectory, '--target', 'sd-cli', 'sd-server', '--parallel', "$Parallel") (Join-Path $BuildDirectory 'compile.log') | Out-Null
    $binaryDirectory = Join-Path $cmakeDirectory 'bin'
    foreach ($name in @('sd-cli.exe', 'sd-server.exe', 'stable-diffusion.dll')) {
        if (-not (Test-Path -LiteralPath (Join-Path $binaryDirectory $name))) { throw "Expected compiled output is absent: $name" }
    }
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    foreach ($file in Get-ChildItem -LiteralPath $BaseRuntimeDirectory -File) {
        if ($file.Name -eq 'wildimage-runtime.json') { continue }
        Copy-ReplacingFile $file.FullName (Join-Path $Destination $file.Name)
    }
    foreach ($name in @('sd-cli.exe', 'sd-server.exe', 'stable-diffusion.dll')) {
        Copy-ReplacingFile (Join-Path $binaryDirectory $name) (Join-Path $Destination $name)
    }
    $licenses = @{
        'thirdparty/libwebp/COPYING' = 'libwebp-COPYING.txt'
        'thirdparty/libwebp/PATENTS' = 'libwebp-PATENTS.txt'
        'thirdparty/libwebm/LICENSE.TXT' = 'libwebm-LICENSE.txt'
        'thirdparty/libwebm/PATENTS.TXT' = 'libwebm-PATENTS.txt'
        'thirdparty/oniguruma/COPYING' = 'oniguruma-COPYING.txt'
        'thirdparty/utf8proc/LICENSE.md' = 'utf8proc-LICENSE.md'
    }
    foreach ($relative in $licenses.Keys) { Copy-ReplacingFile (Join-Path $SourceDirectory $relative) (Join-Path $Destination $licenses[$relative]) }
    Invoke-Checked (Join-Path $Destination 'sd-cli.exe') @('--help') (Join-Path $BuildDirectory 'help.log') | Out-Null
    $files = @(Get-ChildItem -LiteralPath $Destination -File | Where-Object { $_.Name -ne 'wildimage-runtime.json' } | ForEach-Object { File-Record $_.FullName $_.Name })
    $manifest = @{ provider = 'stable-diffusion.cpp'; source = 'https://github.com/leejet/stable-diffusion.cpp'; revision = $runtimeRevision; upstream_revision = $sourceRevision; base_release = $baseManifest.release; base_archive_sha256 = $baseManifest.archive_sha256; features = @{ conditioning_cache = $true; ram_budget = $true; gpu_memory_retry_exit = 75; ram_budget_exit = 76 }; build = $preparation; files = $files }
    $manifest | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $Destination 'wildimage-runtime.json') -Encoding utf8
    Write-Output "WildImage custom runtime prepared: $Destination"
    Write-Output 'Released GGML/CUDA DLLs were reused unchanged. No model was downloaded or loaded.'
} finally {
    Get-ChildItem Env: | Where-Object { -not $originalEnvironment.ContainsKey($_.Name) } | ForEach-Object { [Environment]::SetEnvironmentVariable($_.Name, $null, 'Process') }
    foreach ($name in $originalEnvironment.Keys) { [Environment]::SetEnvironmentVariable($name, $originalEnvironment[$name], 'Process') }
}
