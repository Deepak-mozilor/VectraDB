#!/usr/bin/env bash
set -euo pipefail
# Publish crates in dependency order. This script will attempt to publish and
# will continue even if a crate version already exists (it warns and moves on).
CRATES=(
  "src/components"
  "src/search"
  "src/storage"
  "src/api"
  "src/chunkers"
  "src/server"
)

for c in "${CRATES[@]}"; do
  echo "\n=== Packaging $c ==="
  (cargo package --manifest-path "$c/Cargo.toml") || { echo "cargo package failed for $c"; exit 1; }
  echo "=== Publishing $c ==="
  (cargo publish --manifest-path "$c/Cargo.toml") || {
    echo "cargo publish failed for $c - it might already be published or there was an error. Skipping.";
  }
  echo "Waiting 8 seconds for crates.io index to update..."
  sleep 8
done

echo "Done." 
