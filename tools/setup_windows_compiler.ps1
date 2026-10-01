# Initialize Microsoft's native developer shell without a JavaScript action.
$ErrorActionPreference = 'Stop'
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$installation = & $vswhere -latest -products '*' -version '[17.0,18.0)' `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if ($LASTEXITCODE -or !$installation) { throw 'Visual Studio 2022 C++ tools were not found' }
$before = [Environment]::GetEnvironmentVariables('Process')
& (Join-Path $installation 'Common7/Tools/Launch-VsDevShell.ps1') `
    -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
if ($env:VSCMD_ARG_TGT_ARCH -ne 'x64') { throw 'The x64 compiler environment was not initialized' }
Get-Command cl.exe -ErrorAction Stop | Out-Null

# Each Actions step gets a fresh process. Persist only the environment changes
# made by the developer shell, including PATH (which retains setup-python).
if ($env:GITHUB_ENV) {
    foreach ($entry in [Environment]::GetEnvironmentVariables('Process').GetEnumerator()) {
        $name = [string]$entry.Key
        $value = [string]$entry.Value
        if ($before[$name] -ceq $value) { continue }
        if ($name -match '^(GITHUB_|RUNNER_|NODE_OPTIONS$)') { continue }
        $delimiter = 'VSENV_' + [Guid]::NewGuid().ToString('N')
        "$name<<$delimiter`n$value`n$delimiter" | Add-Content -LiteralPath $env:GITHUB_ENV -Encoding utf8
    }
}
