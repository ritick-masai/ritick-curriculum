Parse the arguments: $ARGUMENTS

The arguments are quoted strings in this order:
0. batch_id — the batch folder, e.g. `IITP-AIMLPX-ADV-2608`. It must already exist in the repo root
1. module_name — the module the session belongs to (context only, never printed in the pre-read)
2. session_name — the session title
3. topics — the topics to cover, separated by semicolons
4. learning_objectives — the objectives, separated by semicolons
5. setup — `Yes` or `No`
6. setup_text — only when setup is Yes: the setup text or documentation link to paste

If any of arguments 0 to 5 is missing, ask for all missing inputs in one short message. Never guess them.
If setup is `Yes` and there is no setup_text, ask the user to paste it. If setup is `No`, skip the Set Up section.

Execute the following pipeline:

## Step 1 — Find the lecture folder

Run from the repo root (the folder that contains `preread/`, `tools/` and the batch folders):

```bash
python3 tools/resolve_folder.py "<batch_id>" "<session_name>"
```

- It prints the existing lecture folder, e.g. `IITP-AIMLPX-ADV-2608/module-02-core-machine-learning-engineering/lecture-01-the-ml-workflow-and-data-splits`.
- Exit code 2 means the batch or session did not match. Show the `ASK:` line to the user and stop. Never create or rename a folder.
- `folder_path` = the repo root joined with the printed path.
- The folder holds `.gitkeep` until the pre-read is written. Leave `.gitkeep` in place.
- If `{folder_path}/pre-read.md` already exists, ask the user before overwriting it.

## Step 2 — Invoke the creator agent

Spawn the `agent_preread_creator` agent with these inputs:
- module_name (from arg 1)
- session_name (from arg 2)
- topics (from arg 3)
- learning_objectives (from arg 4)
- setup (from arg 5)
- setup_text (from arg 6, only when setup is Yes)
- folder_path (from Step 1)

Wait for it to finish. It saves `pre-read.md` only. No PNG is created. Outcome bullets open with Bloom's verbs from the bank in `preread/guidelines.md`. Any code preview is tested before it stays.

## Step 3 — Invoke the reviewer agent

Spawn the `agent_preread_verifier` agent with:
- preread_path: `{folder_path}/pre-read.md`
- module_name, session_name, topics, learning_objectives and setup (from the arguments)

## Step 4 — Report

Print to the user:
- Path to the saved `pre-read.md`
- Status: Accepted / Rejected, with iterations and fixes applied
- The scenario the pre-read uses
- Remarks (if rejected)
- Anything unverified, such as a pasted link that was not opened
