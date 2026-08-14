param(
    [string]$SourceRoot = "D:\Hung_MBL\monarch_behavior_windows\new_cohort_C01-C08",
    [string]$RepoRoot = "C:\Users\MBLuser\Documents\GitHub\basler_caterpillar_recorder"
)

$ErrorActionPreference = "Stop"

$SourceRoot = (Resolve-Path $SourceRoot).Path
$RepoRoot = (Resolve-Path $RepoRoot).Path
$Dest = Join-Path $SourceRoot "source_data"

Write-Host "Creating portable timeline source-data bundle:"
Write-Host "  Dataset: $SourceRoot"
Write-Host "  Repo:    $RepoRoot"
Write-Host "  Output:  $Dest"

New-Item -ItemType Directory -Force -Path $Dest | Out-Null

# These files are the frozen inputs needed by the portable plotting wrapper.
$required = @(
    (Join-Path $SourceRoot "recording_coverage.csv"),
    (Join-Path $SourceRoot "behavior_events_used.csv"),
    (Join-Path $SourceRoot "cropped_by_caterpillar\leaf_feeding\feeding_events.csv"),
    (Join-Path $SourceRoot "cropped_by_caterpillar\motion_energy\motion_states.csv"),
    (Join-Path $RepoRoot "plot_recording_timeline.py"),
    (Join-Path $RepoRoot "analysis_timing.py"),
    (Join-Path $RepoRoot "plot_timeline_from_source_data.py")
)

foreach ($path in $required) {
    if (-not (Test-Path $path)) {
        throw "Required file is missing: $path"
    }
}

# Frozen scientific inputs.
Copy-Item (Join-Path $SourceRoot "recording_coverage.csv") `
    (Join-Path $Dest "recording_coverage.csv") -Force

Copy-Item (Join-Path $SourceRoot "behavior_events_used.csv") `
    (Join-Path $Dest "behavior_events_used.csv") -Force

$eventsSource = Join-Path $SourceRoot "behavior_events_source.json"
if (Test-Path $eventsSource) {
    Copy-Item $eventsSource (Join-Path $Dest "behavior_events_source.json") -Force
}

Copy-Item (Join-Path $SourceRoot "cropped_by_caterpillar\leaf_feeding\feeding_events.csv") `
    (Join-Path $Dest "feeding_events.csv") -Force

Copy-Item (Join-Path $SourceRoot "cropped_by_caterpillar\motion_energy\motion_states.csv") `
    (Join-Path $Dest "motion_states.csv") -Force

# Keep the current rendered figure as a visual reference.
$currentFigure = Join-Path $SourceRoot "recording_behavior_timeline.png"
if (Test-Path $currentFigure) {
    Copy-Item $currentFigure `
        (Join-Path $Dest "recording_behavior_timeline_reference.png") -Force
}

# Freeze the exact plotting implementation used at archive time.
Copy-Item (Join-Path $RepoRoot "plot_recording_timeline.py") `
    (Join-Path $Dest "plot_recording_timeline.py") -Force

Copy-Item (Join-Path $RepoRoot "analysis_timing.py") `
    (Join-Path $Dest "analysis_timing.py") -Force

Copy-Item (Join-Path $RepoRoot "plot_timeline_from_source_data.py") `
    (Join-Path $Dest "plot_timeline_from_source_data.py") -Force

# Minimal environment requirements for this portable plotting bundle.
@"
numpy>=1.26
matplotlib>=3.9,<4
tzdata>=2024.1
"@ | Set-Content -Encoding UTF8 (Join-Path $Dest "requirements_timeline.txt")

# Save the full active Python environment as an additional reproducibility record.
try {
    & python -m pip freeze | Set-Content -Encoding UTF8 `
        (Join-Path $Dest "python_environment_freeze.txt")
} catch {
    Write-Warning "Could not save pip freeze: $_"
}

# Save repo commit if this is a git checkout.
$commit = "unknown"
try {
    $commit = (& git -C $RepoRoot rev-parse HEAD).Trim()
} catch {
    Write-Warning "Could not determine git commit."
}
$commit | Set-Content -Encoding UTF8 (Join-Path $Dest "repo_commit.txt")

# Human-readable reproduction notes.
$archiveTime = (Get-Date).ToString("o")
@"
Portable source-data bundle for the monarch caterpillar behavior timeline.

Archived:
$archiveTime

Original dataset root:
$SourceRoot

Repository root at archive time:
$RepoRoot

Repository commit:
$commit

Frozen plot inputs:
- recording_coverage.csv
- behavior_events_used.csv
- feeding_events.csv
- motion_states.csv

Code snapshot:
- plot_recording_timeline.py
- analysis_timing.py
- plot_timeline_from_source_data.py

Optional provenance/reference:
- behavior_events_source.json
- recording_behavior_timeline_reference.png
- python_environment_freeze.txt

Recreate the figure:
    cd "$Dest"
    python plot_timeline_from_source_data.py

The recreated figure is written to:
    $Dest\recording_behavior_timeline.png

This portable wrapper intentionally reads recording_coverage.csv directly.
It therefore does NOT require the original MP4 files or timestamp sidecars
just to recreate this frozen timeline figure.
"@ | Set-Content -Encoding UTF8 (Join-Path $Dest "README_REPRODUCE.txt")

# Hash the archived files so future copies can be checked for integrity.
$hashFiles = Get-ChildItem -Path $Dest -File |
    Where-Object { $_.Name -ne "SHA256SUMS.txt" } |
    Sort-Object Name

$hashLines = foreach ($file in $hashFiles) {
    $hash = (Get-FileHash -Algorithm SHA256 $file.FullName).Hash.ToLower()
    "$hash  $($file.Name)"
}
$hashLines | Set-Content -Encoding UTF8 (Join-Path $Dest "SHA256SUMS.txt")

Write-Host ""
Write-Host "Archive complete."
Write-Host "Portable source data: $Dest"
Write-Host ""
Write-Host "To recreate:"
Write-Host "  cd `"$Dest`""
Write-Host "  python plot_timeline_from_source_data.py"
