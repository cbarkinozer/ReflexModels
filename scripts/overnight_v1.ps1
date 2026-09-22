param(
    [Parameter(Mandatory=$true)][int]$PreviousQueuePid
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
$queueDirectory = Join-Path $projectRoot 'results\overnight_v1'
New-Item -ItemType Directory -Path $queueDirectory -Force | Out-Null
Set-Location $projectRoot

function Run-Stage {
    param([string]$Name, [string[]]$Arguments)
    "$(Get-Date -Format o) starting $Name" | Add-Content (Join-Path $queueDirectory 'status.log')
    $process = Start-Process -FilePath $pythonPath -ArgumentList $Arguments -WorkingDirectory $projectRoot -RedirectStandardOutput (Join-Path $queueDirectory "$Name.stdout.log") -RedirectStandardError (Join-Path $queueDirectory "$Name.stderr.log") -WindowStyle Hidden -Wait -PassThru
    if ($process.ExitCode -ne 0) {
        "$(Get-Date -Format o) failed $Name exit=$($process.ExitCode)" | Add-Content (Join-Path $queueDirectory 'status.log')
        throw "$Name failed"
    }
    "$(Get-Date -Format o) completed $Name" | Add-Content (Join-Path $queueDirectory 'status.log')
}

while (Get-Process -Id $PreviousQueuePid -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 20
}

Run-Stage -Name 'qwen_v1_real_smoke' -Arguments @(
    '-u', 'scripts\train_v1_decision_model.py', '--model-path', 'data\raw\models\Qwen-Qwen3-0.6B-Base',
    '--data', 'data\prepared\v1\massive_tr_tiny.jsonl', '--output-dir', 'results\qwen3_0_6b_v1_head_smoke',
    '--freeze-backbone', '--epochs', '1', '--batch-size', '1', '--max-length', '128', '--num-threads', '6'
)
Run-Stage -Name 'qwen_v1_head_pilot' -Arguments @(
    '-u', 'scripts\train_v1_decision_model.py', '--model-path', 'data\raw\models\Qwen-Qwen3-0.6B-Base',
    '--data', 'data\prepared\v1\massive_tr_pilot.jsonl', '--output-dir', 'results\qwen3_0_6b_v1_head_pilot',
    '--freeze-backbone', '--epochs', '3', '--batch-size', '4', '--max-length', '128', '--num-threads', '6'
)
