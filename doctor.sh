#!/usr/bin/env bash
# Check that this machine can run the skill and that Claude Code can reach this checkout.
# Strictly read-only. Bash 3.2 compatible. Lives at the repository root; `bench.py init` does not copy it.
# The fix messages quote commands in backticks; none of it is meant to expand.
# shellcheck disable=SC2016
set -euo pipefail

usage() {
    printf '%s\n' \
        'Usage: doctor.sh [-h | --help]' \
        '' \
        '  Required     python3 3.9 or newer; a missing item makes the exit status 1' \
        '  Claude Code  whether the skill is installed under ~/.claude/skills; never changes the exit status' \
        '' \
        'Strictly read-only. HOME is honored.'
}

if [[ $# -gt 0 ]]; then
    case "$1" in
        -h | --help) usage; exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
fi

unset CDPATH
case "$0" in
    */*) script_dir=${0%/*} ;;
    *) script_dir=. ;;
esac
skill_root=$(cd "$script_dir" && pwd -P)
home=${HOME:-}
[[ -n "$home" ]] || { printf '%s\n' 'HOME is not set.' >&2; exit 1; }

# report STATUS ITEM [FIX]: one line, the status word padded to 8 columns.
report() {
    if [[ $# -ge 3 ]]; then printf '%-8s %s — %s\n' "$1" "$2" "$3"; else printf '%-8s %s\n' "$1" "$2"; fi
}

missing=0
printf '%s\n' 'Required'
py_fix='install Python 3.9 or newer, for example from python.org or your package manager'
if ! command -v python3 > /dev/null 2>&1; then
    report MISSING python3 "$py_fix"
    missing=1
elif py_version=$(python3 -c 'import sys; v = sys.version_info; print("%d.%d.%d" % v[:3]); sys.exit(v < (3, 9))' 2> /dev/null); then
    report ok "python3 $py_version"
else
    report MISSING "python3 ${py_version:-(broken)}, 3.9 or newer is required" "$py_fix"
    missing=1
fi

printf '\n%s\n' 'Claude Code'
installed="$home/.claude/skills/reference-benchmark"
if [[ ! -f "$installed/SKILL.md" ]]; then
    report note 'skill reference-benchmark is not installed' "run \`mkdir -p ~/.claude/skills && ln -s \"$skill_root\" ~/.claude/skills/reference-benchmark\`, or clone it as shown under Install in README.md"
elif [[ "$(cd "$installed" && pwd -P)" == "$skill_root" ]]; then
    report ok 'skill reference-benchmark: ~/.claude/skills/reference-benchmark resolves to this checkout'
else
    report note 'skill reference-benchmark' "the installed copy is a different directory ($(cd "$installed" && pwd -P)); changes in this checkout do not reach it"
fi

printf '\n'
if [[ "$missing" -eq 0 ]]; then
    printf 'doctor: ready\n'
else
    printf 'doctor: %d required item(s) missing\n' "$missing"
    exit 1
fi
