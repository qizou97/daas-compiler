#!/usr/bin/env bash
# UserPromptSubmit hook for daas-compiler workflow shortcuts.
# Detects keyword triggers in the user's prompt and injects context.

INPUT=$(cat)
PROMPT=$(echo "$INPUT" | jq -r '.tool_input.prompt // ""' 2>/dev/null)

SKILL_MD="/home1/zouqi/codes/daas-compiler/skills/daas-compiler/SKILL.md"
RELEASE_SH="/home1/zouqi/codes/daas-compiler/release.sh"

if echo "$PROMPT" | grep -qE '本地更新|更新到本地'; then
    printf '{"hookSpecificOutput":{"hookEventName":"UserPromptSubmit","additionalContext":"[daas-compiler hook] 检测到本地更新指令。请立即用 Bash tool 执行：claude plugin update daas-compiler，完成后告知用户结果。"}}\n'

elif echo "$PROMPT" | grep -qiE '\brelease\b|发布'; then
    CURRENT=$(grep '^version:' "$SKILL_MD" 2>/dev/null | awk '{print $2}')
    MAJOR=$(echo "$CURRENT" | cut -d. -f1)
    MINOR=$(echo "$CURRENT" | cut -d. -f2)
    PATCH=$(echo "$CURRENT" | cut -d. -f3)
    NEXT="${MAJOR}.${MINOR}.$((PATCH + 1))"
    printf '{"hookSpecificOutput":{"hookEventName":"UserPromptSubmit","additionalContext":"[daas-compiler hook] 检测到 release/发布 指令。当前版本: %s，默认下一版本: %s（patch bump）。若用户未指定版本号则用 %s，否则用用户指定版本。确认后运行 %s <version>，该脚本自动更新4处版本号(SKILL.md/pyproject.toml/marketplace.json/plugin.json)并 commit+tag+push。"}}\n' "$CURRENT" "$NEXT" "$NEXT" "$RELEASE_SH"
fi
