# Going public: checklist

The repo is private. Before switching it to public (decided 2026-09-29: scrub at go-public time, after in-flight
branches merge, one force-push):

1. **Remove from git history** (`git filter-repo --invert-paths --path ...`, then force-push main and delete stale
   remote branches):
   - `bench/results/prefill-depth/greedy-8ea46e9d0/`, `bench/results/prefill-depth/greedy-pf-v6/`: 128-token model
     continuations of private sessions. Also scan every other `bench/results/**/greedy*/` and `*.out` for generated
     text or prompts from private sessions (`git log --all --stat -- 'bench/results/**/greedy*'`).
   - `bench/replay/corpus/manifest.json`: local paths (/home/ric), working directories, session ids.
   - `bench/replay/corpus/system_prompt.txt`, `bench/replay/corpus/tools.json`: third-party (oh-my-pi) text with
     unverified redistribution terms; the system prompt also lists local skills.
2. **Review** `deploy/draft-vocab/qwen3.8-27b.txt` (token ids ranked from private sessions: low risk; keep or
   regenerate from a public corpus).
3. **Apache-2.0 §4(b)**: add a one-line "modified from tensorninja/ninfer-4090" header to each modified file in
   `apps/web` (list in THIRD_PARTY_NOTICES.md).
4. Make sure the raw sessions, KLD windows and proxy captures are still gitignored (`git check-ignore`), then
   `git grep -nE '/home/ric|riclewis'` on the rewritten history for leftovers.
5. Switch the GitHub repo to public; check that the README renders (mermaid, SVG charts).
