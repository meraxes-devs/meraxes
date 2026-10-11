#!/bin/bash
# Format the C/C++/CUDA sources under src/ with clang-format 14 and the repository's
# .clang-format. CI checks formatting with this script and the same clang-format version
# (.github/workflows/clang-format-check.yml), so run it before pushing.
#
# Usage: tools/format.sh [--check [--cached]] [FILE...]
#   Without FILEs it formats every tracked *.c, *.h, *.cu, *.cpp and *.hh under src/
#   (the mlog submodule is not included).
#   --check   changes nothing: prints the diff clang-format would make and exits 1 if any.
#   --cached  checks the staged versions of the files instead (used by .githooks/pre-commit).
#
# clang-format 14 is taken from $CLANG_FORMAT, clang-format-14 or clang-format, whichever
# is version 14 first. Other versions format some constructs differently, so they're
# refused. Install it with `pipx install clang-format==14.0.6` (or pip into a venv).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

check=false
cached=false
if [ "${1:-}" = --check ]; then
  check=true
  shift
  if [ "${1:-}" = --cached ]; then
    cached=true
    shift
  fi
fi

cf=""
for c in "${CLANG_FORMAT:-}" clang-format-14 clang-format; do
  if [ -n "$c" ] && command -v "$c" > /dev/null && "$c" --version | grep -q 'version 14\.'; then
    cf=$c
    break
  fi
done
if [ -z "$cf" ]; then
  echo "clang-format 14 not found; CI uses 14.0.6. Install it with: pipx install clang-format==14.0.6" >&2
  exit 2
fi

if [ $# -gt 0 ]; then
  files=("$@")
else
  mapfile -t files < <(git ls-files -- 'src/*.c' 'src/*.h' 'src/*.cu' 'src/*.cpp' 'src/*.hh')
fi

if ! $check; then
  "$cf" -i "${files[@]}"
  exit 0
fi

contents() {
  if $cached; then git show ":$1"; else cat -- "$1"; fi
}

status=0
for f in "${files[@]}"; do
  if ! diff -u --label "$f" --label "$f (clang-format)" \
    <(contents "$f") <(contents "$f" | "$cf" --assume-filename="$f"); then
    status=1
  fi
done
if [ $status -ne 0 ]; then
  echo "These files aren't formatted; run tools/format.sh to fix them." >&2
fi
exit $status
