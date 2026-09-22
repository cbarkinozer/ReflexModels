param(
    [Parameter(Mandatory=$true)][int]$BerturkWorkerPid
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
$resultsRoot = Join-Path $projectRoot 'results'
$primaryResult = Join-Path $resultsRoot 'berturk_tr_checkpointed\result.json'
$queueDirectory = Join-Path $resultsRoot 'overnight_queue'
New-Item -ItemType Directory -Path $queueDirectory -Force | Out-Null
Set-Location $projectRoot

function Run-Stage {
    param([string]$Name, [string[]]$Arguments)
    $outputPath = Join-Path $queueDirectory "$Name.stdout.log"
    $errorPath = Join-Path $queueDirectory "$Name.stderr.log"
    "$(Get-Date -Format o) starting $Name" | Add-Content (Join-Path $queueDirectory 'status.log')
    $process = Start-Process -FilePath $pythonPath -ArgumentList $Arguments -WorkingDirectory $projectRoot -RedirectStandardOutput $outputPath -RedirectStandardError $errorPath -WindowStyle Hidden -Wait -PassThru
    if ($process.ExitCode -ne 0) {
        "$(Get-Date -Format o) failed $Name exit=$($process.ExitCode)" | Add-Content (Join-Path $queueDirectory 'status.log')
        throw "$Name failed; inspect $errorPath"
    }
    "$(Get-Date -Format o) completed $Name" | Add-Content (Join-Path $queueDirectory 'status.log')
}

while (Get-Process -Id $BerturkWorkerPid -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 20
}
if (-not (Test-Path -LiteralPath $primaryResult)) {
    throw 'Primary BERTurk run exited without result.json; inspect its log and saved checkpoints'
}

Run-Stage -Name 'validate_berturk' -Arguments @('scripts\validate_fixed_label_result.py', 'results\berturk_tr_checkpointed')
if (-not (Test-Path 'results\qwen3_0_6b_v0_negation\result.json')) {
Run-Stage -Name 'v0_negation' -Arguments @(
    'scripts\evaluate_v0_negation.py', '--model-path', 'data\raw\models\Qwen-Qwen3-0.6B-Base',
    '--output-dir', 'results\qwen3_0_6b_v0_negation', '--num-threads', '6'
)
}
if (-not (Test-Path 'results\tokenizer_fertility_massive_local.json')) {
Run-Stage -Name 'tokenizer_fertility' -Arguments @(
    'scripts\benchmark_tokenizers.py', '--config', 'configs\tokenizer_fertility_massive_local.json',
    '--output', 'results\tokenizer_fertility_massive_local.json'
)
}
if (-not (Test-Path 'results\berturk_tr_sensitivity_smoke\result.json')) {
Run-Stage -Name 'berturk_sensitivity_smoke' -Arguments @(
    '-u', 'scripts\run_fixed_label_baseline.py', '--config', 'configs\baselines\massive_fixed_label.example.json',
    '--run', 'berturk_tr', '--model-path', 'data\raw\models\dbmdz-bert-base-turkish-cased',
    '--overlap-policy', 'drop_train_dev_overlapping_test', '--output-dir', 'results\berturk_tr_sensitivity_smoke',
    '--epochs', '1', '--batch-size', '4', '--max-length', '64', '--num-threads', '4',
    '--limit-train', '8', '--limit-dev', '8', '--limit-test', '8'
)
}
if (-not (Test-Path 'results\berturk_tr_sensitivity\result.json')) {
Run-Stage -Name 'berturk_sensitivity' -Arguments @(
    '-u', 'scripts\run_fixed_label_baseline.py', '--config', 'configs\baselines\massive_fixed_label.example.json',
    '--run', 'berturk_tr', '--model-path', 'data\raw\models\dbmdz-bert-base-turkish-cased',
    '--overlap-policy', 'drop_train_dev_overlapping_test', '--output-dir', 'results\berturk_tr_sensitivity',
    '--epochs', '3', '--batch-size', '8', '--max-length', '128', '--num-threads', '6'
)
}
Run-Stage -Name 'validate_sensitivity' -Arguments @('scripts\validate_fixed_label_result.py', 'results\berturk_tr_sensitivity')
