---
description: Snapshot all EvoX state to a timestamped git commit
---

Create an EvoX state checkpoint by running these steps:

1. Stage all state files (shared + all GPU-specific):
```bash
git add evox/state_gpu*.json evox/current_strategy_gpu*.md evox/population.json evox/strategies.json
```

2. Stage candidate directories if they exist:
```bash
git add candidates/ 2>/dev/null || true
```

3. Stage archived strategies:
```bash
git add evox/strategies/ 2>/dev/null || true
```

4. Create a descriptive checkpoint commit:
```bash
git commit -m "evox: manual checkpoint - GPU ${EVOX_GPU:-0}, window #$(uv run evox/state_manager.py get --key window_count 2>/dev/null | grep -o '[0-9]*' || echo '?'), $(uv run python -c "import json; p=json.load(open('evox/population.json')); print(len(p))" 2>/dev/null || echo '?') candidates"
```

5. Report what was committed:
```bash
git log --oneline -1
git diff --stat HEAD~1
```

Tell the user what was checkpointed (number of files, latest candidate count, current window, GPU index).
