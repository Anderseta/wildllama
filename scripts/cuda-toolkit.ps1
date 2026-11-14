# Toolkit lookup must work for both installer and extracted redistributable
# layouts. A long-running build host can retain CUDA_PATH after uninstallation.
function Get-WildLlamaCudaCompilerVersion([string]$Directory) {
    $compiler = Join-Path $Directory 'bin/nvcc.exe'
    if (-not (Test-Path -LiteralPath $compiler -PathType Leaf)) { return $null }
    try { $output = (& $compiler --version 2>&1 | Out-String) } catch { return $null }
    if ($LASTEXITCODE -ne 0 -or $output -notmatch 'release (\d+\.\d+), V([0-9.]+)') { return $null }
    return [version]$Matches[2]
}

function Find-WildLlamaCudaToolkit {
    [CmdletBinding()]
    param(
        [string]$Directory,
        [string]$Version,
        [int]$Major = 0,
        [switch]$Optional,
        [string[]]$Candidates
    )
    if ($Version -and $Version -notmatch '^\d+\.\d+(\.\d+)?$') { throw 'Use a CUDA version such as 13.5.' }
    $paths = @()
    if ($Directory) {
        $paths = @($Directory)
    } elseif ($PSBoundParameters.ContainsKey('Candidates')) {
        $paths = $Candidates
    } else {
        $names = if ($Version) { @(('CUDA_PATH_V' + $Version.Replace('.', '_')), 'CUDA_PATH', 'CUDA_HOME') } else { @('CUDA_PATH', 'CUDA_HOME') }
        foreach ($scope in @('Process', 'User', 'Machine')) {
            foreach ($name in $names) {
                $paths += [Environment]::GetEnvironmentVariable($name, $scope)
            }
        }
        $compiler = Get-Command nvcc.exe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($compiler) { $paths += Split-Path (Split-Path $compiler.Source -Parent) -Parent }
        $base = Join-Path $env:ProgramFiles 'NVIDIA GPU Computing Toolkit/CUDA'
        $paths += @(Get-ChildItem -LiteralPath $base -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match '^v\d+\.\d+$' } |
            Sort-Object { [version]$_.Name.Substring(1) } -Descending | Select-Object -ExpandProperty FullName)
    }
    foreach ($path in @($paths | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Select-Object -Unique)) {
        # CUDA configuration is a directory, not an executable or shell command.
        if (-not [IO.Path]::IsPathRooted($path) -or -not (Test-Path -LiteralPath $path -PathType Container)) { continue }
        $actual = Get-WildLlamaCudaCompilerVersion $path
        if (-not $actual -or ($Major -and $actual.Major -ne $Major)) { continue }
        if ($Version) {
            $requested = [version]$Version
            if ($actual.Major -ne $requested.Major -or $actual.Minor -ne $requested.Minor -or
                ($requested.Build -ge 0 -and $actual.Build -ne $requested.Build)) { continue }
        }
        return [IO.Path]::GetFullPath($path)
    }
    if ($Optional -and -not $Directory) { return $null }
    throw "CUDA compiler not found for directory '$Directory', version '$Version', major '$Major'. Supply an installed toolkit with -CudaDirectory."
}
