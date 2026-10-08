#!/usr/bin/env bash
# Sync the vault into this Quartz site and build it.
#
# Publishes by default: Stas does not want a preview step (2026-10-08, "pls just publish,
# i dont care"). Pass --no-push to build without publishing. The vault audit below still
# blocks the build on dangling links.
#
# Paths are derived from this script's location. The previous version hardcoded
# /home/ubuntu paths from the OpenClaw VPS, which no longer exists.
set -euo pipefail

SITE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VAULT="${VAULT:-$(cd "$SITE/../university-vault" && pwd)}"

[ -d "$VAULT/Courses" ] || { echo "No vault at $VAULT. Set VAULT=/path/to/university-vault" >&2; exit 1; }

echo "Vault: $VAULT"
echo "Site:  $SITE"

# Other machines push here too. Start from the latest site so the push at the end lands.
if [ "${1:-}" != "--no-push" ]; then
  git -C "$SITE" pull --rebase --autostash -q
fi

# Course homes and flashcard pages are generated from the notes. Regenerate them so card
# counts and lecture lists never go stale, and say so if that changed the vault.
if [ -f "$VAULT/docs/superpowers/tools/course_pages.py" ]; then
  echo
  echo "Generating course pages..."
  python3 "$VAULT/docs/superpowers/tools/course_pages.py" | sed 's/^/  /'
  if [ -n "$(git -C "$VAULT" status --porcelain -- Courses)" ]; then
    echo "  note: the vault has uncommitted changes under Courses/. Commit them so the vault matches the site."
  fi
fi

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

# public/ is the deploy artifact. Cloudflare Pages serves exactly what is committed
# there, so any file git ignores under public/ is a page that silently never goes live.
# Not hypothetical: public/ sat in .gitignore from 2026-09-08 to 2026-09-11. Already-tracked
# pages kept updating, new ones were never added, and eight notes were built and published
# to nowhere while the index pages linked to them. Fail loudly instead.
IGNORED=$(git -C "$SITE" status --porcelain --ignored=matching -- public | grep '^!!' || true)
if [ -n "$IGNORED" ]; then
  echo >&2
  echo "public/ contains git-ignored files. These pages will NOT deploy:" >&2
  echo "$IGNORED" | head -10 >&2
  echo "Take public/ out of .gitignore." >&2
  exit 1
fi
echo "  $(find "$SITE/public" -name '*.html' | wc -l) pages built, none ignored"

if [ "${1:-}" != "--no-push" ]; then
  echo
  echo "Publishing..."
  git add -A
  git commit -m "Update notes $(date +%Y-%m-%d)" --allow-empty
  git push
  echo "Pushed. Cloudflare Pages will deploy."
else
  echo
  echo "Built, not published (--no-push). Preview it with:"
  echo "    cd $SITE && npx quartz build --serve"
fi
