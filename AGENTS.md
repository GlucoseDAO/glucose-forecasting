# Agent conventions

## CLI naming

For new command-line interfaces, avoid long hyphenated executable names. Prefer a concise root Typer command with noun/verb subcommands:

```text
glucose forecast
glucose train
glucose models
glucose train --model sugarone
glucose train --backend neuralforecast --eval holdout
```

Use kebab-case only for multi-word option names when needed. When a command selects an
ML evaluation protocol, use precise terms such as `--eval holdout` or `--eval
cross-val`; reserve “workflow” for multi-step orchestration. Preserve existing command
names and behavior for compatibility.

## Personalization boundary

Do not edit `scripts/personalization/` unless the user explicitly requests personalization work. It is maintained independently; consume its existing public imports from new package code rather than moving or rewriting it.
