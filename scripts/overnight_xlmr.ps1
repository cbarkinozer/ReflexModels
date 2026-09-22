param(
    [Parameter(Mandatory=$true)][int]$PreviousQueuePid
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
$queueDirectory = Join-Path $projectRoot 'results\overnight_xlmr'
New-Item -ItemType Directory -Path $queueDirectory -Force | Out-Null
Set-Location $projectRoot

function Run-Stage {
    param([string]$Name, [string[]]$Arguments)
    "$(Get-Date -Format o) starting $Name" | Add-Content (Join-Path $queueDirectory 'status.log')
    $outputPath = Join-Path $queueDirectory "$Name.stdout.log"
    $errorPath = Join-Path $queueDirectory "$Name.stderr.log"
    $process = Start-Process -FilePath $pythonPath -ArgumentList $Arguments -WorkingDirectory $projectRoot -RedirectStandardOutput $outputPath -RedirectStandardError $errorPath -WindowStyle Hidden -Wait -PassThru
    if ($process.ExitCode -ne 0) {
        "$(Get-Date -Format o) failed $Name exit=$($process.ExitCode)" | Add-Content (Join-Path $queueDirectory 'status.log')
        throw "$Name failed"
    }
    "$(Get-Date -Format o) completed $Name" | Add-Content (Join-Path $queueDirectory 'status.log')
}

while (Get-Process -Id $PreviousQueuePid -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 20
}
if (-not (Test-Path -LiteralPath (Join-Path $projectRoot 'results\berturk_tr_sensitivity\result.json'))) {
    throw 'Previous queue did not produce the BERTurk sensitivity result'
}

foreach ($language in @('tr', 'en')) {
    $runName = "xlmr_$language"
    Run-Stage -Name "${runName}_smoke" -Arguments @(
        '-u', 'scripts\run_fixed_label_baseline.py', '--config', 'configs\baselines\massive_fixed_label.example.json',
        '--run', $runName, '--model-path', 'data\raw\models\FacebookAI-xlm-roberta-base',
        '--overlap-policy', 'report_only', '--output-dir', "results\${runName}_smoke",
        '--epochs', '1', '--batch-size', '4', '--max-length', '64', '--num-threads', '4',
        '--limit-train', '8', '--limit-dev', '8', '--limit-test', '8'
    )
    Run-Stage -Name $runName -Arguments @(
        '-u', 'scripts\run_fixed_label_baseline.py', '--config', 'configs\baselines\massive_fixed_label.example.json',
        '--run', $runName, '--model-path', 'data\raw\models\FacebookAI-xlm-roberta-base',
        '--overlap-policy', 'report_only', '--output-dir', "results\$runName",
        '--epochs', '3', '--batch-size', '8', '--max-length', '128', '--num-threads', '6'
    )
    Run-Stage -Name "validate_$runName" -Arguments @('scripts\validate_fixed_label_result.py', "results\$runName")
}
