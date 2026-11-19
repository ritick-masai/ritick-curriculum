# Pre-reads & Lecture Scripts Pipeline

## File Structure

```text
<sme-name>-curriculum/              ← one repo per SME / CC (e.g. ritick-curriculum)
├── README.md
├── templates/
│   ├── pre-read-template.md
│   └── lecture-script-template.md
├── <batch-id-1>/                   ← one folder per batch (batch name from INI sheet)
│   ├── module-01-<topic>/
│   │   ├── lecture-01-<topic>/
│   │   │   ├── pre-read.md
│   │   │   └── lecture-script.md
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

## Naming Rules

Lowercase and hyphens only, numbered modules and lectures, and the same two file names everywhere (`pre-read.md`, `lecture-script.md`).

- 📁 One repo per SME/CC (`<your-name>-curriculum`), edit access for everyone
- 📂 One folder per batch (same ID as the INI sheet) → module folders → lecture folders
- 📄 Each lecture folder has 2 files: `pre-read.md` and `lecture-script.md`
