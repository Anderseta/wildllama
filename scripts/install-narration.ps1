param(
    [Parameter(Mandatory)][string]$WildllamaRoot,
    [string]$ModelsDir = $env:OLLAMA_MODELS,
    [string]$Python = 'python',
    [string]$Uv = 'uv'
)
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath($WildllamaRoot)
if (!$ModelsDir) { $ModelsDir = Join-Path $root 'models' }
$runtime = Join-Path $root 'runtime/narration'
$voice = Join-Path $ModelsDir 'voice'
$library = Join-Path $root 'lib/narration'
Get-Command ffmpeg -ErrorAction Stop | Out-Null
$env:UV_CACHE_DIR = Join-Path $root 'runtime/uv-cache'
$env:HF_HOME = Join-Path $root 'runtime/hf-cache'
$env:TEMP = Join-Path $runtime 'temp'
$env:TMP = $env:TEMP
[void][IO.Directory]::CreateDirectory($env:TEMP)
foreach ($spec in @(
    @('tts','qwen-tts==0.1.1','Qwen3-TTS-12Hz-1.7B-Base','fd4b254389122332181a7c3db7f27e918eec64e3'),
    @('aligner','qwen-asr==0.0.6','Qwen3-ForcedAligner-0.6B','c7cbfc2048c462b0d63a45797104fc9db3ad62b7')
)) {
    $dir = Join-Path $runtime $spec[0]
    $py = Join-Path $dir 'Scripts/python.exe'
    if (!(Test-Path -LiteralPath $py)) {
        & $Uv venv --python $Python $dir
        if ($LASTEXITCODE) { throw 'Falha ao preparar o Python. Use Python 3.12 x64.' }
    }
    & $Uv pip install --python $py --index-url https://download.pytorch.org/whl/cu128 'torch==2.8.0+cu128' 'torchaudio==2.8.0+cu128' 'torchvision==0.23.0+cu128'
    if ($LASTEXITCODE) { throw 'Falha ao instalar PyTorch.' }
    & $Uv pip install --python $py $spec[1]
    if ($LASTEXITCODE) { throw "Falha ao instalar $($spec[1])." }
    & $py -c "import sys; from huggingface_hub import snapshot_download; snapshot_download('Qwen/'+sys.argv[1], revision=sys.argv[2],local_dir=sys.argv[3],allow_patterns=['*.safetensors','*.json','*.txt'])" $spec[2] $spec[3] (Join-Path $voice $spec[2])
    if ($LASTEXITCODE) { throw "Falha ao baixar $($spec[2]). Execute novamente para retomar." }
}
[void][IO.Directory]::CreateDirectory($library)
Copy-Item -LiteralPath (Join-Path $PSScriptRoot '../narration/worker.py') -Destination (Join-Path $library 'worker.py')
Write-Output "Narração instalada. Modelos em $voice."
