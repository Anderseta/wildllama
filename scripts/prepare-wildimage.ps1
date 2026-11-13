[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Destination,
    [Parameter(Mandatory=$true)][string]$DownloadDirectory,
    [string]$CudaDirectory = $env:CUDA_PATH
)
$ErrorActionPreference = 'Stop'
$revision = 'b9d7176bf88fe465576e0aafbe47ba253ffcfc2c'
$release = 'master-1091-b9d7176'
$archiveName = 'sd-master-b9d7176-bin-win-cuda12-x64.zip'
$expectedHash = '36ab0884ab3b09e7f062cc936544503364d43a84231e7671d47bdedbb5369292'
$uri = "https://github.com/leejet/stable-diffusion.cpp/releases/download/$release/$archiveName"
function Assert-DataPath([string]$Path) {
    $absolute = [IO.Path]::GetFullPath($Path)
    if ([IO.Path]::GetPathRoot($absolute) -ieq 'C:\') { throw 'Image payloads and downloads must be outside C:.' }
    $ancestor = $absolute
    while (-not (Test-Path -LiteralPath $ancestor)) {
        $parent = Split-Path -Parent $ancestor
        if (-not $parent -or $parent -eq $ancestor) { break }
        $ancestor = $parent
    }
    if (Test-Path -LiteralPath $ancestor) {
        $item = Get-Item -LiteralPath $ancestor
        if ($item.PSObject.Methods.Name -contains 'ResolveLinkTarget') {
            $target = $item.ResolveLinkTarget($true)
            if ($target -and [IO.Path]::GetPathRoot($target.FullName) -ieq 'C:\') { throw 'Image destination resolves to C:.' }
        }
    }
    return $absolute
}
$Destination = Assert-DataPath $Destination
$DownloadDirectory = Assert-DataPath $DownloadDirectory
New-Item -ItemType Directory -Force -Path $Destination,$DownloadDirectory | Out-Null
$archive = Join-Path $DownloadDirectory $archiveName
if (-not (Test-Path -LiteralPath $archive)) { Invoke-WebRequest -Uri $uri -OutFile $archive }
$hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash.ToLowerInvariant()
if ($hash -ne $expectedHash) { throw "Runtime archive SHA256 mismatch: $hash" }
Expand-Archive -LiteralPath $archive -DestinationPath $Destination -Force

# Only vendor CUDA runtimes may be copied from the existing toolkit. Image ggml
# and stable-diffusion DLLs exclusively come from the pinned image archive.
if (-not $CudaDirectory) { throw 'CUDA_PATH or -CudaDirectory is required; no toolkit is installed automatically.' }
$vendor = @()
foreach ($name in @('cudart64_12.dll','cublas64_12.dll','cublasLt64_12.dll')) {
    $source = Join-Path $CudaDirectory "bin/$name"
    if (-not (Test-Path -LiteralPath $source)) { throw "Required CUDA runtime not found: $source" }
    Copy-Item -LiteralPath $source -Destination (Join-Path $Destination $name) -Force
    $vendor += @{name=$name;source=$source;sha256=(Get-FileHash -Algorithm SHA256 -LiteralPath $source).Hash.ToLowerInvariant()}
}
$eula = Join-Path $CudaDirectory 'EULA.txt'
if (Test-Path -LiteralPath $eula) { Copy-Item -LiteralPath $eula -Destination (Join-Path $Destination 'CUDA-EULA.txt') -Force }
$files = @()
Get-ChildItem -LiteralPath $Destination -File | Where-Object {$_.Extension -in '.exe','.dll'} | ForEach-Object {
    $files += @{name=$_.Name;size=$_.Length;sha256=(Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()}
}
$manifest = @{provider='stable-diffusion.cpp';source='https://github.com/leejet/stable-diffusion.cpp';revision=$revision;release=$release;archive_url=$uri;archive_sha256=$hash;cuda_vendor_libraries=$vendor;files=$files}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $Destination 'wildimage-runtime.json') -Encoding utf8
Write-Output "WildImage runtime prepared: $Destination"
Write-Output 'This script does not install a toolkit, download models, or load the GPU.'
