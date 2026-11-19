# Pre-reads & Lecture Scripts Pipeline

## File Structure

```text
<sme-name>-curriculum/              ← one repo per SME / CC (e.g. ritick-curriculum)
├── README.md
├── .claude/                        ← Claude Code commands and agents (run from this folder)
│   ├── commands/
│   │   ├── create-pre_read.md
│   │   └── create-lecture-script.md
│   └── agents/
│       ├── agent_preread_creator.md
│       ├── agent_preread_verifier.md
│       ├── agent_lecture_creator.md
│       └── agent_lecture_verifier.md
├── preread/                        ← pre-read rules (guidelines, SKILL, agent + command master copies)
├── lecture/                        ← lecture-script rules (guidelines, terms, playbook, SKILL, agents, commands, tools/)
├── tools/
│   └── resolve_folder.py           ← finds a session's lecture folder inside a batch
├── templates/
│   ├── pre-read-template.md
│   └── lecture-script-template.md
├── curriculum/
│   └── <module>.csv                ← session titles, topics, LOs (source for the lecture folders)
├── examples/
│   ├── preread/                    ← three finished pre-reads (reference only)
│   └── lecture/                    ← three finished lecture scripts with images/data (reference only)
├── <batch-id-1>/                   ← one folder per batch (batch name from INI sheet)
│   ├── module-01-<topic>/
│   │   ├── lecture-01-<topic>/
│   │   │   ├── pre-read.md
│   │   │   ├── lecture-script.md
│   │   │   ├── images/             ← only when the lecture script uses charts
│   │   │   └── data/               ← only when the lecture script uses a dataset
│   │   ├── lecture-02-<topic>/
│   │   │   ├── pre-read.md
│   │   │   └── lecture-script.md
│   │   └── ...
│   ├── module-02-<topic>/
│   │   └── ...
│   └── ...
├── <batch-id-2>/
├── <batch-id-3>/
├── <batch-id-4>/
└── <batch-id-5>/
```

An empty lecture folder holds a `.gitkeep` until its files are written.

## Naming Rules

Lowercase and hyphens only, numbered modules and lectures, and the same two file names everywhere (`pre-read.md`, `lecture-script.md`).

- 📁 One repo per SME/CC (`<your-name>-curriculum`), edit access for everyone
- 📂 One folder per batch (same ID as the INI sheet) → module folders → lecture folders
- 📄 Each lecture folder has 2 files: `pre-read.md` and `lecture-script.md` (plus `images/` and `data/` when a script needs them)

## Create a pre-read or a lecture script

Requirements: Claude Code (signed in) and Python 3. Lecture scripts that run code also need `pandas`, `scipy`, `scikit-learn`, `matplotlib`; SQL sessions need a local MySQL 8+.

Run Claude Code from the repo root. The batch and lecture folder must already exist; the commands write into it and never create folders.

```
/create-pre_read "BATCH-ID" "Module Name" "Session Name" "Topic 1; Topic 2" "Objective 1; Objective 2" "No"
/create-pre_read "BATCH-ID" "Module Name" "Session Name" "Topics" "Objectives" "Yes" "paste the setup text or link here"

/create-lecture-script "BATCH-ID" "Module Name" "Session Name" "Topic 1; Topic 2" "LO 1; LO 2" "No" ["path/to/curriculum.csv"]
/create-lecture-script "BATCH-ID" "Module Name" "Session Name" "Topics" "LOs" "Yes" ["path/to/curriculum.csv"] "Announcement text"
```

- `Module Name`, `Session Name`, `Topics` and `Objectives`/`LOs` come from the row in `curriculum/<module>.csv`.
- The session name must match the lecture folder: `The ML Workflow and Data Splits` → `lecture-01-the-ml-workflow-and-data-splits`.
- The curriculum CSV defaults to `curriculum/core-machine-learning-engineering.csv`.

Each command runs a creator agent, then a verifier agent that fixes issues in place.

| Command | Writes | Rules |
|---|---|---|
| `/create-pre_read` | `<batch>/<module>/<lecture>/pre-read.md` | `preread/guidelines.md`, `templates/pre-read-template.md` |
| `/create-lecture-script` | `<batch>/<module>/<lecture>/lecture-script.md` (+ `images/`, `data/`) | `lecture/guidelines.md`, `lecture/terms.md`, `lecture/playbook.md`, `templates/lecture-script-template.md` |

Helper scripts:

```
python3 tools/resolve_folder.py "BATCH-ID" "Session Name"                            # prints the lecture folder
python3 lecture/tools/curriculum_context.py curriculum/<module>.csv "Session Name"   # what the batch completed
python3 lecture/tools/check_script.py <lecture-script.md> --los "LO 1;LO 2" --announcements no --module "Module" --session "Session"
```

## Keep the team in sync

- Change rules in one place: `preread/guidelines.md`, `lecture/guidelines.md`, `lecture/terms.md` and the two files in `templates/`.
- After editing `preread/agents/*.md` or `lecture/agents/*.md`, copy them over the matching file in `.claude/agents/` (`creator.md` → `agent_*_creator.md`, `reviewer.md` → `agent_*_verifier.md`). The two copies must stay identical.
- Share changes through git. Do not let each person edit a private copy.
