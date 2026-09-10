#!/usr/bin/env bash
# Sync the vault into this Quartz site and build it.
#
# Publishing is deliberately NOT the default. Run with --push only when Stas has
# looked at the preview and said go.
#
# Paths are derived from this script's location. The previous version hardcoded
# /home/ubuntu paths from the OpenClaw VPS, which no longer exists.
set -euo pipefail

SITE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VAULT="${VAULT:-$(cd "$SITE/../university-vault" && pwd)}"

[ -d "$VAULT/Courses" ] || { echo "No vault at $VAULT. Set VAULT=/path/to/university-vault" >&2; exit 1; }

echo "Vault: $VAULT"
echo "Site:  $SITE"

# The vault is the source of truth. Audit before publishing anything from it.
if [ -f "$VAULT/docs/superpowers/tools/vault_audit.py" ]; then
  echo
  echo "Auditing the vault..."
  if ! python3 "$VAULT/docs/superpowers/tools/vault_audit.py" | tail -1 | grep -q "RESULT: PASS"; then
    echo "Vault audit FAILED. Fix the dangling links before publishing." >&2
    exit 1
  fi
  echo "  audit passed"
fi

echo
echo "Copying content..."
rm -rf "$SITE/content/Concepts" "$SITE/content/Courses" "$SITE/content/Assets"
cp -r "$VAULT/Concepts" "$SITE/content/"
cp -r "$VAULT/Courses" "$SITE/content/"
[ -d "$VAULT/Assets" ] && cp -r "$VAULT/Assets" "$SITE/content/"
echo "  $(find "$SITE/content" -name '*.md' | wc -l) notes"

echo
echo "Building..."
cd "$SITE"
npx quartz build 2>&1 | tail -4

if [ "${1:-}" = "--push" ]; then
  echo
  echo "Publishing..."
  git add -A
  git commit -m "Update notes $(date +%Y-%m-%d)" --allow-empty
  git push
  echo "Pushed. Cloudflare Pages will deploy."
else
  echo
  echo "Built, not published. Preview it with:"
  echo "    cd $SITE && npx quartz build --serve"
  echo "Then re-run with --push to publish."
fi
