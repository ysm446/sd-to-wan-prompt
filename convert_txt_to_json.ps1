param(
    [Parameter(Mandatory = $false)]
    [string]$TargetDir = ".",
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"

# Normalize incoming path text (some shell invocations may include stray quotes).
if ($null -eq $TargetDir) {
    $TargetDir = ""
}
$TargetDir = $TargetDir.Trim()
$TargetDir = $TargetDir.Trim('"').Trim()
if ([string]::IsNullOrWhiteSpace($TargetDir)) {
    $TargetDir = "."
}

try {
    $TargetDir = [System.IO.Path]::GetFullPath($TargetDir)
} catch {
    Write-Host "[ERROR] Invalid target path: $TargetDir"
    exit 1
}

function Get-SectionText {
    param(
        [string]$Text,
        [string]$StartMarker,
        [string[]]$EndMarkers
    )

    $start = $Text.IndexOf($StartMarker, [System.StringComparison]::Ordinal)
    if ($start -lt 0) { return "" }
    $start = $start + $StartMarker.Length

    $end = $Text.Length
    foreach ($marker in $EndMarkers) {
        $idx = $Text.IndexOf($marker, $start, [System.StringComparison]::Ordinal)
        if ($idx -ge 0 -and $idx -lt $end) {
            $end = $idx
        }
    }

    return $Text.Substring($start, $end - $start).Trim()
}

if (-not (Test-Path -LiteralPath $TargetDir)) {
    Write-Host "[ERROR] Folder not found: $TargetDir"
    exit 1
}

$target = (Resolve-Path -LiteralPath $TargetDir).Path
$txtFiles = Get-ChildItem -LiteralPath $target -Filter *.txt -File -Recurse

$converted = 0

foreach ($txt in $txtFiles) {
    $text = Get-Content -LiteralPath $txt.FullName -Raw -Encoding UTF8

    if ($text -notlike "*=== Original Prompt ===*" -or $text -notlike "*=== Generated WAN Prompt ===*") {
        Write-Host ("[SKIP] {0} (not WAN txt format)" -f $txt.Name)
        continue
    }

    $original = Get-SectionText -Text $text -StartMarker "=== Original Prompt ===" -EndMarkers @("=== Additional Instruction ===", "=== Generated WAN Prompt ===")
    $additional = Get-SectionText -Text $text -StartMarker "=== Additional Instruction ===" -EndMarkers @("=== Generated WAN Prompt ===")
    $generated = Get-SectionText -Text $text -StartMarker "=== Generated WAN Prompt ===" -EndMarkers @()
    if ($null -eq $additional) { $additional = "" }
    if ($null -eq $original) { $original = "" }
    if ($null -eq $generated) { $generated = "" }
    $additional = [string]$additional
    $original = [string]$original
    $generated = [string]$generated

    $imagePath = [System.IO.Path]::ChangeExtension($txt.FullName, ".png")
    $imageFilename = [System.IO.Path]::GetFileName($imagePath)
    $jsonPath = [System.IO.Path]::ChangeExtension($txt.FullName, ".json")

    if ((Test-Path -LiteralPath $jsonPath) -and (-not $Overwrite)) {
        Write-Host ("[SKIP] {0} already exists" -f $jsonPath)
        continue
    }

    $payload = [ordered]@{
        image_filename = $imageFilename
        image_path = $imagePath
        metadata = [ordered]@{
            path = $imagePath
            filename = $imageFilename
            size = $null
            prompt = $original
            negative_prompt = ""
            settings = @{}
        }
        prompt = $generated
        additional_instruction = $additional
        original_prompt = $original
    }

    $payload | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding UTF8
    Write-Host ("[OK] {0} -> {1}" -f $txt.Name, [System.IO.Path]::GetFileName($jsonPath))
    $converted += 1
}

Write-Host ""
Write-Host ("Converted: {0} file(s)" -f $converted)
