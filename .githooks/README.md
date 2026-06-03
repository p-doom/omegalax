# Git hooks

Tracked hooks for this repo. Enable them once per clone:

```sh
git config core.hooksPath .githooks
```

(This applies to all worktrees that share the clone's git dir.)

## `pre-commit`

Blocks a commit if any **staged** Python file is not `ruff`-formatted, and
prints the exact command to fix it. Requires [`uv`](https://docs.astral.sh/uv/)
on `PATH` (it runs `uvx ruff …`; no separate install needed).

- Formatting rules live in `pyproject.toml` (`[tool.ruff]`, line-length 100).
- The ruff version is pinned inside the hook so it matches the repo-wide
  formatting baseline.
- Format the whole repo manually with:
  `uvx ruff format omegalax scripts tests`
- Bypass for a single commit (discouraged) with `git commit --no-verify`.
