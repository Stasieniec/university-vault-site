#!/usr/bin/env bash
set -e

VAULT="/home/ubuntu/university-vault"
SITE="/home/ubuntu/university-vault-site"

echo "🔄 Syncing vault → site..."
rm -rf "$SITE/content/Concepts" "$SITE/content/Courses" "$SITE/content/Assets"
cp -r "$VAULT/Concepts" "$SITE/content/"
cp -r "$VAULT/Courses" "$SITE/content/"
cp -r "$VAULT/Assets" "$SITE/content/" 2>/dev/null || true
cp "$VAULT/VAULT-INSTRUCTIONS.md" "$SITE/content/" 2>/dev/null || true

cd "$SITE"

echo "🔨 Building..."
npx quartz build 2>&1 | tail -5

echo "📤 Pushing to GitHub..."
git add -A
git commit -m "Update notes $(date +%Y-%m-%d)" --allow-empty
git push

echo "✅ Done — Cloudflare Pages will deploy automatically."
