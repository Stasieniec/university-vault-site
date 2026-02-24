#!/usr/bin/env bash
# Sync vault content to Quartz site and deploy
set -e

VAULT="$HOME/Desktop/University/university-vault"
SITE="$HOME/Desktop/University/university-vault-site"

echo "🔄 Syncing vault → site..."
rm -rf "$SITE/content/Concepts" "$SITE/content/Courses" "$SITE/content/Assets"
cp -r "$VAULT/Concepts" "$SITE/content/"
cp -r "$VAULT/Courses" "$SITE/content/"
cp -r "$VAULT/Assets" "$SITE/content/" 2>/dev/null || true
cp "$VAULT/VAULT-INSTRUCTIONS.md" "$SITE/content/" 2>/dev/null || true

cd "$SITE"

# Quick local build check
echo "🔨 Building..."
npx quartz build 2>&1 | tail -3

echo "📤 Pushing to GitHub..."
git add -A
git commit -m "Update notes $(date +%Y-%m-%d)" --allow-empty
git push

echo "✅ Done — Cloudflare Pages will deploy automatically."
