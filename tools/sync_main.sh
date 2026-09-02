#!/usr/bin/env bash
# Regenerate the onboardee-facing `main` branch from the `solutions` branch.
#
# Maintainers edit code, docs, and tests on `solutions` only. This script copies the
# tree onto `main`, strips every SOLUTION block into a TODO, and commits. Never edit
# generated files on `main` by hand; the next sync would overwrite the edit.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -n "$(git status --porcelain)" ]; then
  echo "working tree is not clean; commit or stash first" >&2
  exit 1
fi

git checkout -q solutions
python3 tools/make_blanks.py --check >/dev/null && { echo "solutions branch has no SOLUTION blocks; nothing to strip?" >&2; }
git checkout -q main
git checkout solutions -- .
python3 tools/make_blanks.py --apply
python3 tools/make_blanks.py --check >/dev/null && ok=1 || ok=0
if [ "$ok" != 1 ]; then echo "blanking failed: markers remain" >&2; exit 1; fi
git add -A
if git diff --cached --quiet; then
  echo "main is already up to date"
else
  git commit -q -m "Sync from solutions ($(git rev-parse --short solutions))"
  echo "main updated: $(git rev-parse --short main)"
fi
