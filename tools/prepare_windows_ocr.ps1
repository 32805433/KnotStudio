# Build the native OCR engine and preserve all installed dependency notices.
$ErrorActionPreference = 'Stop'
$project = Split-Path $PSScriptRoot -Parent
$manifest = Join-Path $project 'packaging/vcpkg'
$baseline = (Get-Content "$manifest/vcpkg.json" | ConvertFrom-Json).'builtin-baseline'
$checkout = Join-Path $project 'build/vcpkg'
$installed = Join-Path $project 'build/vcpkg_installed'
if (!(Test-Path "$checkout/.git")) {
    git clone --filter=blob:none https://github.com/microsoft/vcpkg.git $checkout
    if ($LASTEXITCODE) { throw 'vcpkg clone failed' }
}
git -C $checkout checkout --detach $baseline
if ($LASTEXITCODE) { throw 'vcpkg checkout failed' }
& "$checkout/bootstrap-vcpkg.bat" -disableMetrics
if ($LASTEXITCODE) { throw 'vcpkg bootstrap failed' }
$env:VCPKG_DISABLE_METRICS = '1'
& "$checkout/vcpkg.exe" install --triplet x64-windows-static-md `
    "--x-manifest-root=$manifest" "--x-install-root=$installed" `
    "--overlay-triplets=$manifest/triplets"
if ($LASTEXITCODE) { throw 'OCR build failed' }
$exe = Join-Path $installed 'x64-windows-static-md/tools/tesseract/tesseract.exe'
if (!(Test-Path $exe)) { throw 'OCR executable missing' }
Write-Output "OCR build ready: $exe"
