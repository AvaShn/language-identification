# step1_clean_all.ps1
# Run from project root (where the "data" folder exists)

$inRoot  = "data\raw"
$outRoot = "data\clean"

# Target audio settings
$targetSR = 16000

# Silence trimming (tweak if needed)
$startThresh = "-35dB"
$stopThresh  = "-35dB"
$startSilence = "0.2"
$stopSilence  = "0.2"

# Ensure output root exists
New-Item -ItemType Directory -Force -Path $outRoot | Out-Null

# Find ALL mp3 files under data\raw (recursive)
Get-ChildItem -Path $inRoot -Filter *.mp3 -Recurse | ForEach-Object {

    $inFile = $_.FullName

    # Build relative path under raw (e.g., es\male\file.mp3)
    $relPath = $inFile.Substring((Resolve-Path $inRoot).Path.Length).TrimStart('\')

    # Change extension to .wav and map to clean folder
    $relOut  = [System.IO.Path]::ChangeExtension($relPath, ".wav")
    $outFile = Join-Path $outRoot $relOut

    # Create output directory if needed
    $outDir = Split-Path $outFile -Parent
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null

    Write-Host "Processing: $relPath"

      $af = "loudnorm=I=-16:TP=-1.5:LRA=11"

      ffmpeg -y -hide_banner -loglevel error -i "$inFile" `
        -map a:0 -vn -sn -dn `
        -ac 1 -ar $targetSR `
        -af "$af" `
        -c:a pcm_s16le `
        "$outFile"


    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $relPath" -ForegroundColor Red
    }
}

Write-Host "Done. Cleaned WAVs are in data\clean (same folder structure)." -ForegroundColor Green
