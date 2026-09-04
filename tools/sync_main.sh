#!/usr/bin/env bash
# Regenerate the onboardee-facing `main` branch from the `solutions` branch.
#
# Maintainers edit code, docs, and tests on `solutions` only. This script replaces
# main's tree with solutions', strips every SOLUTION block into a TODO, and commits.
# Never edit generated files on `main` by hand; the next sync would overwrite the edit.
#
# Run it from a checkout with a clean working tree. It leaves you on `main`; push with
#     git push origin main
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -n "$(git status --porcelain)" ]; then
  echo "working tree is not clean; commit or stash first" >&2
  exit 1
fi

git rev-parse --verify -q solutions >/dev/null || { echo "no local 'solutions' branch" >&2; exit 1; }
git rev-parse --verify -q main >/dev/null || { echo "no local 'main' branch" >&2; exit 1; }

git checkout -q solutions
if python3 tools/make_blanks.py --check >/dev/null; then
  echo "solutions has no SOLUTION blocks; nothing to strip?" >&2
  exit 1
fi

git checkout -q main

# Make the index and working tree exactly match solutions, without moving main's HEAD.
# This is the step that has to *delete* files too: a plain `git checkout solutions -- .`
# only adds and updates paths that exist on solutions, so anything deleted there would
# survive on main forever.
git read-tree -u --reset solutions

python3 tools/make_blanks.py --apply
if ! python3 tools/make_blanks.py --check >/dev/null; then
  echo "blanking failed: markers remain" >&2
  exit 1
fi

git add -A
if git diff --cached --quiet; then
  echo "main is already up to date"
else
  git commit -q -m "Sync from solutions ($(git rev-parse --short solutions))"
  echo "main updated: $(git rev-parse --short main)"
fi
