#!/usr/bin/env bash
# Usage: ./release.sh <new_version>
# Example: ./release.sh 0.7.11
#
# Bumps version in all 4 places, commits, tags, and pushes.
# Run AFTER confirming the skill works locally with:
#   claude plugin update daas-compiler

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
NEW_VER="${1:-}"

if [[ -z "$NEW_VER" ]]; then
    echo "Usage: $0 <new_version>  (e.g. 0.7.11)"
    exit 1
fi

# Validate semver format
if ! [[ "$NEW_VER" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: version must be x.y.z (got: $NEW_VER)"
    exit 1
fi

OLD_VER=$(grep '^version:' "$REPO_DIR/skills/daas-compiler/SKILL.md" | awk '{print $2}')
echo "Bumping $OLD_VER → $NEW_VER"

# 1. SKILL.md frontmatter
sed -i "s/^version: .*/version: $NEW_VER/" "$REPO_DIR/skills/daas-compiler/SKILL.md"

# 2. skills/daas-compiler/pyproject.toml
sed -i "s/^version = \".*\"/version = \"$NEW_VER\"/" "$REPO_DIR/skills/daas-compiler/pyproject.toml"

# 3. .claude-plugin/marketplace.json  (two occurrences)
sed -i "s/\"version\": \"$OLD_VER\"/\"version\": \"$NEW_VER\"/g" "$REPO_DIR/.claude-plugin/marketplace.json"

# 4. .claude-plugin/plugin.json
sed -i "s/\"version\": \"$OLD_VER\"/\"version\": \"$NEW_VER\"/" "$REPO_DIR/.claude-plugin/plugin.json"

# Verify
echo ""
echo "--- version check ---"
grep "^version" "$REPO_DIR/skills/daas-compiler/SKILL.md" "$REPO_DIR/skills/daas-compiler/pyproject.toml"
grep '"version"' "$REPO_DIR/.claude-plugin/marketplace.json" "$REPO_DIR/.claude-plugin/plugin.json"
echo "---------------------"
echo ""

# Commit + tag + push
cd "$REPO_DIR"
git add skills/daas-compiler/SKILL.md \
        skills/daas-compiler/pyproject.toml \
        .claude-plugin/marketplace.json \
        .claude-plugin/plugin.json

git commit -m "release: bump version to v$NEW_VER"
git tag "v$NEW_VER"
git push origin main
git push origin "v$NEW_VER"

echo ""
echo "✓ Released v$NEW_VER"
