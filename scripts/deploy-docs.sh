#!/usr/bin/env bash
# Deploy docs to gh-pages with versioning.
# Merges new version into existing gh-pages content.
# Usage: deploy-docs.sh VERSION [ALIAS] [SITE_DIR] [DEPLOY_DIR]
#   deploy-docs.sh dev          -> publish site/ to dev/
#   deploy-docs.sh 0.12 latest  -> publish site/ to 0.12/ and latest/

set -euo pipefail

VERSION="${1:?Usage: deploy-docs.sh VERSION [ALIAS]}"
ALIAS="${2:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SITE_DIR="${3:-site}"
DEPLOY_DIR="${4:-.deploy-docs-workdir}"

cd "$REPO_ROOT"
mkdir -p "$DEPLOY_DIR"
cd "$DEPLOY_DIR"

if [[ ! -d .git ]]; then
  git init
  REPO_URL="https://github.com/${GITHUB_REPOSITORY:-mlco2/ecologits}.git"
  if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    REPO_URL="https://x-access-token:${GITHUB_TOKEN}@${REPO_URL#https://}"
  fi
  git remote add origin "$REPO_URL"
fi

git fetch origin gh-pages --depth=1 2>/dev/null || true
if git show-ref --verify --quiet refs/remotes/origin/gh-pages 2>/dev/null; then
  git checkout -B gh-pages origin/gh-pages
else
  git checkout --orphan gh-pages
  git rm -rf . 2>/dev/null || true
fi

rm -rf "$VERSION"
cp -r "$REPO_ROOT/$SITE_DIR" "$VERSION"

if [[ -n "$ALIAS" ]]; then
  # mike left the alias as a symlink; replace it with a copy.
  rm -rf "$ALIAS"
  cp -r "$VERSION" "$ALIAS"
fi

# GitHub Pages serves only /404.html for a missing path. Its links are absolute
# /latest/... paths (site_url), so the copy works from the root.
if [[ -f latest/404.html ]]; then
  cp latest/404.html 404.html
fi

VERSIONS_JSON="versions.json"
VERSION="$VERSION" ALIAS="$ALIAS" python3 << 'PYEOF'
import json
import os

version, alias = os.environ["VERSION"], os.environ["ALIAS"]
try:
    with open("versions.json") as f:
        data = json.load(f)
except FileNotFoundError:
    data = []

data = [v for v in data if v["version"] != version]
if alias:
    for v in data:
        v["aliases"] = [a for a in v.get("aliases", []) if a != alias]
entry = {"version": version, "title": version, "aliases": [alias] if alias else []}
# Keep dev first, then releases newest first, as mike did.
data.insert(1 if data and data[0]["version"] == "dev" and version != "dev" else 0, entry)

with open("versions.json", "w") as f:
    json.dump(data, f, indent=2)
PYEOF

cat > index.html << 'HTMLEOF'
<!DOCTYPE html>
<html>
<head>
  <meta http-equiv="refresh" content="0; url=latest/">
  <script>window.location.replace("latest/");</script>
</head>
<body>Redirecting to <a href="latest/">latest documentation</a>...</body>
</html>
HTMLEOF

echo "ecologits.ai" > CNAME
touch .nojekyll

git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git add -A
if git diff --staged --quiet; then
  echo "No changes to deploy"
else
  git commit -m "Deploy docs $VERSION${ALIAS:+ ($ALIAS)}"
  git push origin gh-pages
fi
