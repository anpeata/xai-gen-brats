Write-Host "Configuring repository hooks path to .githooks ..."

git config core.hooksPath .githooks

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to configure core.hooksPath"
    exit 1
}

Write-Host "Hooks installed successfully. Pre-push policy is now active."
