# PowerShell script to publish crates in order
$crates = @(
  "src/components",
  "src/search",
  "src/storage",
  "src/api",
  "src/chunkers",
  "src/server"
)

foreach ($c in $crates) {
  Write-Host "`n=== Packaging $c ==="
  Push-Location $c
  try {
    cargo package
  } catch {
    Write-Warning "cargo package failed for $c. Aborting."
    Pop-Location
    exit 1
  }
  Write-Host "=== Publishing $c ==="
  try {
    cargo publish
  } catch {
    Write-Warning "cargo publish failed for $c. It may already exist or an error occurred. Skipping."
  }
  Pop-Location
  Start-Sleep -Seconds 8
}
Write-Host "Done."
