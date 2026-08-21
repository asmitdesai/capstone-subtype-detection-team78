$ErrorActionPreference = "Continue"

$logDir = "Dataset\Datasets\download_batches\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$failLog = "$logDir\failures.log"

foreach ($n in 1..7) {
    $batchNum = "{0:D2}" -f $n
    $csv = "Dataset\Datasets\download_batches\batch_$batchNum.csv"
    $log = "$logDir\batch_$batchNum.log"

    Write-Host "$(Get-Date -Format o) - starting batch_$batchNum"
    python "$PSScriptRoot\download_tcga_thca_slides.py" $csv *> $log

    # idc_index swallows s5cmd failures internally and still exits 0, so
    # $LASTEXITCODE alone can't be trusted - check the log for s5cmd's own
    # ERROR lines too (network drop, retries exhausted, etc).
    $hasError = ($LASTEXITCODE -ne 0) -or (Select-String -Path $log -Pattern "^ERROR" -Quiet)

    if ($hasError) {
        Add-Content $failLog "$(Get-Date -Format o) - batch_$batchNum FAILED (exit $LASTEXITCODE) - see $log"
        Write-Host "batch_$batchNum FAILED (exit $LASTEXITCODE) - continuing to next batch"
    } else {
        Write-Host "$(Get-Date -Format o) - batch_$batchNum done"
    }
}

Write-Host "All 7 batches attempted. Check $failLog for any failures."
