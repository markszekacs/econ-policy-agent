# run_all.ps1 -- full re-ingest + eval chain; stops at the first failure
$steps = @(
    @("rag\ingest.py", "all"),
    @("eval\run_harness.py", "--parent-too"),
    @("eval\score_arms.py"),
    @("eval\score_arms.py", "--slice", "query_type"),
    @("eval\report.py")
)
foreach ($step in $steps) {
    Write-Host "`n=== py -3.12 $($step -join ' ') ===" -ForegroundColor Cyan
    & py -3.12 @step
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED at: $($step -join ' ') (exit $LASTEXITCODE)" -ForegroundColor Red
        exit 1
    }
}
Write-Host "`nAll steps completed -- see eval\report.md" -ForegroundColor Green