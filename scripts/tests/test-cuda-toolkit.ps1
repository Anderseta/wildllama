$ErrorActionPreference = 'Stop'
. (Join-Path (Split-Path $PSScriptRoot -Parent) 'cuda-toolkit.ps1')
$testRoot = Join-Path ([IO.Path]::GetTempPath()) ('wildllama-cuda-lookup-' + [guid]::NewGuid().ToString('N'))
$originalReader = ${function:Get-WildLlamaCudaCompilerVersion}
function Require([bool]$Condition, [string]$Message) { if (-not $Condition) { throw $Message } }
function Rejects([scriptblock]$Action, [string]$Message) {
    $rejected = $false
    try { & $Action | Out-Null } catch { $rejected = $true }
    Require $rejected $Message
}
try {
    $old = Join-Path $testRoot 'cuda-12.8'
    $current = Join-Path $testRoot 'portable toolkit'
    $missing = Join-Path $testRoot 'uninstalled'
    $runtimeOnly = Join-Path $testRoot 'runtime-only'
    foreach ($path in @($old, $current, $runtimeOnly)) { [void][IO.Directory]::CreateDirectory($path) }
    function Get-WildLlamaCudaCompilerVersion([string]$Directory) {
        if ($Directory -eq $old) { return [version]'12.8.93' }
        if ($Directory -eq $current) { return [version]'13.5.37' }
        return $null
    }
    $paths = @($missing, $runtimeOnly, $current, $old)
    Require ((Find-WildLlamaCudaToolkit -Candidates $paths) -eq $current) 'Stale/runtime-only paths blocked the current toolkit'
    Require ((Find-WildLlamaCudaToolkit -Candidates $paths -Version 13.5) -eq $current) 'Minor version did not match the compiler'
    Require ((Find-WildLlamaCudaToolkit -Candidates $paths -Version 13.5.37) -eq $current) 'Compiler build pin did not match'
    Require ((Find-WildLlamaCudaToolkit -Candidates $paths -Major 12) -eq $old) 'Major version isolation failed'
    Require ((Find-WildLlamaCudaToolkit -Directory $old -Candidates $paths) -eq $old) 'Explicit toolkit was overridden'
    Require ($null -eq (Find-WildLlamaCudaToolkit -Candidates $paths -Version 13.4 -Optional)) 'Missing optional version silently substituted'
    Rejects { Find-WildLlamaCudaToolkit -Candidates $paths -Version 13.5.38 } 'Missing build pin silently substituted'
    Rejects { Find-WildLlamaCudaToolkit -Directory $missing -Candidates $paths -Optional } 'Missing explicit toolkit silently substituted'
    Rejects { Find-WildLlamaCudaToolkit -Directory $old -Version 13.5 } 'Wrong explicit toolkit version accepted'
    Rejects { Find-WildLlamaCudaToolkit -Version '../bad' } 'Malformed version accepted'
    # Exercise candidate collection without modifying persisted environment.
    $savedPath = $env:CUDA_PATH
    $savedHome = $env:CUDA_HOME
    try {
        $env:CUDA_PATH = $missing
        $env:CUDA_HOME = $current
        Require ((Find-WildLlamaCudaToolkit -Version 13.5) -eq $current) 'Versioned environment lookup failed'
        $env:CUDA_PATH = $old
        Require ((Find-WildLlamaCudaToolkit) -eq $old) 'Working process override was ignored'
    } finally { $env:CUDA_PATH = $savedPath; $env:CUDA_HOME = $savedHome }
    Write-Output 'PASS CUDA toolkit lookup: stale paths, extracted layout, major/minor/build pins, explicit overrides and fallback.'
} finally {
    ${function:Get-WildLlamaCudaCompilerVersion} = $originalReader
    # Only these known empty fixture directories are removed, without recursion.
    foreach ($path in @($old, $current, $runtimeOnly, $testRoot)) {
        if ($path -and (Test-Path -LiteralPath $path)) { [IO.Directory]::Delete($path, $false) }
    }
}
