# step2_make_metadata.ps1
$root = "data\clean"
$out  = "metadata.csv"

"filepath,lang,gender" | Out-File -Encoding utf8 $out

Get-ChildItem -Path $root -Filter *.wav -Recurse | ForEach-Object {
    $full = $_.FullName

    # path parts: data\clean\es\male\file.wav
    $rel = $full.Substring((Resolve-Path $root).Path.Length).TrimStart('\')
    $parts = $rel.Split('\')

    $lang = $parts[0]
    $gender = $parts[1]

    # use relative path for portability
    $csvPath = ("data\clean\" + $rel).Replace("\","/")

    "$csvPath,$lang,$gender" | Out-File -Append -Encoding utf8 $out
}

Write-Host "Done: $out created" -ForegroundColor Green
