# Install the optional Windows x64 CUDA runner alongside a WildLlama install.
param([Parameter(Mandatory)][string]$InstallRoot, [string]$Archive)
$ErrorActionPreference = 'Stop'
$release = 'b12728'
$name = "llama-$release-bin-win-cuda-13.5-x64.zip"
$sha256 = 'dc4a26a3b6c8b5c8a5437963e3dd7edd0a6ac4d1e585306291f5bb4d8ee235cd'
$lib = (Resolve-Path -LiteralPath (Join-Path $InstallRoot 'lib/ollama')).Path
if (!(Test-Path -LiteralPath (Join-Path $lib 'llama-server.exe') -PathType Leaf) -or
    !(Test-Path -LiteralPath (Join-Path $lib 'cuda_v13/cudart64_13.dll') -PathType Leaf)) {
    throw 'A Windows x64 WildLlama installation with CUDA 13 is required.'
}
$dest = Join-Path $lib 'inkling2'
$stage = Join-Path $lib ('inkling2.stage-' + [Guid]::NewGuid().ToString('N'))
$null = [IO.Directory]::CreateDirectory($stage)
try {
    if (!$Archive) {
        $Archive = Join-Path $stage $name
        Invoke-WebRequest "https://github.com/ggml-org/llama.cpp/releases/download/$release/$name" -OutFile $Archive
    }
    if ((Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash -ne $sha256) { throw 'Archive checksum mismatch.' }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [IO.Compression.ZipFile]::OpenRead((Resolve-Path -LiteralPath $Archive).Path)
    $files = @('llama-server.exe','llama-server-impl.dll','llama-common.dll','llama.dll','mtmd.dll','ggml.dll','ggml-base.dll','ggml-cuda.dll','libomp.dll','LICENSE-LLVM-OpenMP')
    try {
        $files += @($zip.Entries | Where-Object { $_.FullName -cmatch '^ggml-cpu-[a-z0-9-]+\.dll$' } | ForEach-Object FullName)
        if ($files -notcontains 'ggml-cpu-x64.dll') { throw 'Missing baseline CPU backend.' }
        foreach ($file in $files) {
            $entry = $zip.GetEntry($file)
            if (!$entry) { throw "Missing runtime component $file" }
            [IO.Compression.ZipFileExtensions]::ExtractToFile($entry,(Join-Path $stage $file))
        }
    } finally { $zip.Dispose() }
    $license = Join-Path $lib 'LLAMA_CPP_LICENSE'
    if (Test-Path -LiteralPath $license) { Copy-Item -LiteralPath $license -Destination $stage; $files += 'LLAMA_CPP_LICENSE' }
    $receipt = [ordered]@{release=$release; archive_sha256=$sha256; files=@{}}
    foreach ($file in $files) { $receipt.files[$file] = (Get-FileHash -LiteralPath (Join-Path $stage $file)).Hash.ToLowerInvariant() }
    if (Test-Path -LiteralPath $dest) {
        foreach ($file in $files) {
            if ((Get-FileHash -LiteralPath (Join-Path $dest $file)).Hash -ne $receipt.files[$file]) { throw "Existing Inkling engine differs at $file." }
        }
        Write-Output "Verified installed Inkling engine $release."
    } else {
        $receipt | ConvertTo-Json | Set-Content -Encoding utf8 -LiteralPath (Join-Path $stage 'engine.json')
        if (Test-Path -LiteralPath (Join-Path $stage $name)) { [IO.File]::Delete((Join-Path $stage $name)) }
        [IO.Directory]::Move($stage,$dest)
        Write-Output "Installed Inkling engine $release in $dest."
    }
} finally {
    if (Test-Path -LiteralPath $stage) {
        # This private, flat stage contains only the allowlisted archive entries.
        foreach ($file in [IO.Directory]::EnumerateFiles($stage)) { [IO.File]::Delete($file) }
        [IO.Directory]::Delete($stage)
    }
}
