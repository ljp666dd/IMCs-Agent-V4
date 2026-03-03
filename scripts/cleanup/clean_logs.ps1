param(
    [string]$LogDir = "logs",
    [int]$KeepDays = 14,
    [int]$MaxTotalMB = 500
)

if (-not (Test-Path $LogDir)) {
    Write-Host "Log dir not found: $LogDir"
    exit 0
}

$cutoff = (Get-Date).AddDays(-$KeepDays)
Get-ChildItem -Path $LogDir -File -Recurse | Where-Object { $_.LastWriteTime -lt $cutoff } | Remove-Item -Force -ErrorAction SilentlyContinue

$files = Get-ChildItem -Path $LogDir -File -Recurse | Sort-Object LastWriteTime -Descending
$totalBytes = ($files | Measure-Object -Property Length -Sum).Sum
$maxBytes = $MaxTotalMB * 1MB

if ($totalBytes -gt $maxBytes) {
    $running = $totalBytes
    foreach ($f in ($files | Sort-Object LastWriteTime)) {
        if ($running -le $maxBytes) { break }
        $running -= $f.Length
        Remove-Item -Force -ErrorAction SilentlyContinue -Path $f.FullName
    }
}

Write-Host "Log cleanup done."
