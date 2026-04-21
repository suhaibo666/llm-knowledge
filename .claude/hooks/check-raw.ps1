$rawDir = "e:\97-codes\torch_parallel\llm-knowledge\raw"
$changelog = "e:\97-codes\torch_parallel\llm-knowledge\wiki\changelog.md"

$rawFiles = Get-ChildItem -Path $rawDir -Recurse -File | Select-Object -ExpandProperty Name
$changelogContent = if (Test-Path $changelog) { Get-Content $changelog -Raw } else { "" }

$unprocessed = @()
foreach ($file in $rawFiles) {
    if ($changelogContent -notmatch [regex]::Escape($file)) {
        $unprocessed += $file
    }
}

if ($unprocessed.Count -gt 0) {
    $list = $unprocessed -join ", "
    $message = "UNPROCESSED SOURCES DETECTED: The following files in raw/ have not been ingested into the wiki: $list. Ask the user if they want to ingest any of them."
    Write-Output $message
} else {
    Write-Output "All source files in raw/ have been processed. No pending ingestion."
}
