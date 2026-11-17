# Agent notes for reference-benchmark

This repository is a Claude Code skill: a method for reproducible reference benchmarks
([SKILL.md](SKILL.md)), a metrics catalog and a small standard-library harness (`scripts/`). The
repository root is the skill directory, so every tracked file is installed on every user's machine.
This file is the contract for every agent and human working in this checkout. It holds the rules and
says where to read; the detail lives in the pages it links.

## Rules (read first)

1. **Standard library only.** Python 3.9 or newer; no third-party imports and no dependency files
   (`requirements.txt`, `pyproject.toml`, ...).
2. **Scripts work in place and as a copy.** `bench.py init` copies `bench.py`, every
   `scripts/benchlib_*.py`, `textdiff.py` and `calibrate.py` into the user's project. A new helper named
   `benchlib_*.py` is copied automatically; any other new script must be added to the list in `cmd_init`
   (`scripts/bench.py`) and to `INIT_HELP`, and may import only files that are copied. A copy must not
   point back to the skill by absolute path (pinned by `test_init_copies_toolkit_without_absolute_paths_and_refuses_overwrite`).
3. **History is private and never committed.** Run directories are 0700 and files 0600
   (`scripts/benchlib_util.py`); `history/` stays in `.gitignore`, and `init` adds it to the user's.
4. **Fixtures, templates and examples are synthetic or fictional.** No real data, no real names,
   addresses, paths or host names in any tracked file.
5. **SKILL.md is the product.** Its frontmatter description is loaded into every session of every user:
   keep it tight. SKILL.md "Bundled tools", the README section "What you get" and `python3 scripts/bench.py -h`
   must agree with what the scripts really do.
6. **Nothing dev-only at the root.** Everything tracked ships to users, so no bulk, and no
   `.claude/skills/reference-benchmark` link inside the repository: it would point at the root and loop.
7. **Green before commit:** `make test`. Tests run in temporary directories; they never touch the real
   `~/.claude`, a real history or the network.
8. **English everywhere:** files, comments, commit messages, issues and pull requests. The only German
   is the trigger phrases in the SKILL.md description and the Unicode strings in `test_textdiff.py`.
   The skill itself answers in the user's language, as SKILL.md says.
9. **Clean up after yourself:** temp directories, scratch run histories, `__pycache__` that you created
   and background processes you started are gone before you hand over.

## Which job are you here for?

### A. Installing or using the skill

Two install forms, both in [README.md](README.md#install): clone into `~/.claude/skills/reference-benchmark`
(every project) or into `<project>/.claude/skills/reference-benchmark` (one project). Never overwrite,
delete or replace an existing directory or link there; if the path exists, stop and ask the owner.
Run `make doctor` in a checkout; it is read-only and prints the fix for each missing piece. Then follow
[SKILL.md](SKILL.md) for the benchmark itself. If the project already has a harness, extend it instead
of running `init`. Helper and verifier agents get dry-run or stub only; they never start the real driver.

### B. Changing the skill

- **One behavior, several places.** A change to what the tools do is done when the code, its tests under
  `scripts/tests/`, SKILL.md "Bundled tools", README "What you get" and `bench.py -h` agree. Update the
  test count in the SKILL.md self-test line when it changes.
- **Scripts come with tests.** Every script has its suite; a changed behavior changes its test.
- **The method lives in SKILL.md, by phase.** Do not duplicate it in README.md; the README stays the
  short version.
- **Scope.** Change what was asked. Do not rewrite neighbouring pages or rules in passing; name what
  you noticed instead.

## Where to read

| When you | Read |
| --- | --- |
| run or plan a benchmark | [SKILL.md](SKILL.md): the method, by phase, and "Bundled tools" |
| pick metrics for an output type | [references/metrics-catalog.md](references/metrics-catalog.md) |
| need the harness contract (`bench.json` keys, commands, holdout guard) | `python3 scripts/bench.py -h`, `<command> -h`, `sweep --explain` |
| change a script and need to know what pins its behavior | [scripts/tests/](scripts/tests/) |
| install the skill or use the harness without Claude | [README.md](README.md) |

## Commands

```bash
make help      # list the targets
make doctor    # read-only: python3 3.9 or newer, and whether Claude Code can find this skill
make test      # unit tests (python3 -m unittest discover -s scripts/tests, about 80 s), then doctor.sh checks
```

## Agent host integrations

Claude Code reads `CLAUDE.md`, which is a symlink to this file. Codex, Cursor and other tools that know
`AGENTS.md` read this file directly. Keep the substance here; never turn `CLAUDE.md` into a copy.
