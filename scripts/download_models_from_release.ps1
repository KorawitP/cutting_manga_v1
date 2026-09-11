#Requires -Version 5.1
<#
  โหลดโมเดล production ของ cutting_manga_v1 กลับจาก GitHub Release + ตรวจ SHA256
  Release: cutting-models-v1 (KorawitP/cutting_manga_v1)
  ไฟล์: cnn_v5.pth -> ./cnn_v5.pth
        best_content_filter.pth -> ./tools/content_filter_cli/models/best_content_filter.pth
  ใช้: powershell scripts/download_models_from_release.ps1
#>
$ErrorActionPreference = "Stop"

$Repo = "KorawitP/cutting_manga_v1"
$Tag = "cutting-models-v1"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

$Files = @(
  @{ Name = "cnn_v5.pth"; Dest = (Join-Path $Root "cnn_v5.pth") },
  @{ Name = "best_content_filter.pth"; Dest = (Join-Path $Root "tools\content_filter_cli\models\best_content_filter.pth") }
)

$tmp = Join-Path ([IO.Path]::GetTempPath()) ("cutting-models-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $tmp -Force | Out-Null
try {
  & gh release download $Tag --repo $Repo --dir $tmp --clobber
  if ($LASTEXITCODE -ne 0) { throw "gh release download ล้มเหลว" }

  $sums = @{}
  foreach ($line in (Get-Content (Join-Path $tmp "SHA256SUMS.txt"))) {
    if ($line -match "^([0-9a-fA-F]{64})\s+(.+)$") { $sums[$Matches[2].Trim()] = $Matches[1].ToLower() }
  }

  foreach ($f in $Files) {
    $src = Join-Path $tmp $f.Name
    if (!(Test-Path -LiteralPath $src)) { throw "ไม่พบ $($f.Name) ใน Release" }
    $actual = (Get-FileHash -LiteralPath $src -Algorithm SHA256).Hash.ToLower()
    if ($sums.ContainsKey($f.Name) -and $actual -ne $sums[$f.Name]) {
      throw "$($f.Name) SHA ไม่ตรง (ได้ $actual, ควรเป็น $($sums[$f.Name]))"
    }
    $destDir = Split-Path -Parent $f.Dest
    if (!(Test-Path -LiteralPath $destDir)) { New-Item -ItemType Directory -Path $destDir -Force | Out-Null }
    Copy-Item -LiteralPath $src -Destination $f.Dest -Force
    Write-Output ("OK: {0} -> {1}" -f $f.Name, $f.Dest)
  }
}
finally {
  Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
}
Write-Output "เสร็จ: โมเดลครบ 2 ไฟล์"
