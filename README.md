# Torrent Naming

Organize downloaded movies and TV episodes into a consistent media library. The command-line organizer reads release names, identifies titles and episode numbers, looks up metadata, and creates a plan that you can review before applying.

It can use a local OpenAI-compatible language model (such as LM Studio) to interpret filenames. If the model is unavailable, it falls back to filename heuristics. It looks up production language and release metadata through Wikipedia and IMDb's suggestion endpoint when internet access is available.

## What it does

- Scans a source folder and its subfolders for video and subtitle files.
- Organizes movies and series into language-based folders.
- Matches subtitles to videos and gives them corresponding names.
- Writes a CSV plan so you can review or edit destinations.
- Supports copying or moving files, dry runs, and applying an edited CSV.

Example output:

```text
Media/
├── Movies/
│   └── English/
│       └── Inception (2010)/
│           ├── Inception (2010).mkv
│           └── Inception (2010).en.srt
└── Web Series/
    └── Hindi/
        └── Special Ops (2020)/
            └── Special Ops S02/
                └── Special Ops S02E01.mkv
```

The organizer recognizes common video formats (`mkv`, `mp4`, `avi`, `mov`, `wmv`, `flv`, `m4v`, `webm`, `ts`) and subtitle formats (`srt`, `sub`, `vtt`, `idx`, `ass`, `ssa`).

## Setup

Use Python 3 and create a virtual environment from the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

To install everything used by the script and notebook:

```bash
python -m pip install -r requirements.txt
```

This includes the notebook's ML and MLflow dependencies, which can take time and disk space to install. For the command-line organizer alone, its direct dependencies are smaller:

```bash
python -m pip install openai pandas requests beautifulsoup4 tqdm rich
```

`openai` is optional for the script: without it, title extraction uses the built-in heuristics. Metadata lookup uses Wikipedia and IMDb over the internet; if those services are unavailable, the language is inferred from a clear language tag in the release name when possible, or set to `Unknown`.

## Configure a local model (optional)

The script expects an OpenAI-compatible chat completions endpoint. For LM Studio:

1. Load a model in LM Studio and start its local server.
2. Note the server URL and the model identifier exposed by that server.
3. Pass them with `--base-url` and `--model` when running the organizer.

The script defaults to `http://192.168.0.119:1234/v1` and `google/gemma-4-12b`; set these flags to match your own server. The default API key is `NA`; pass `--api-key` if your endpoint requires a different value. If the endpoint cannot be reached, the script logs a warning and continues with heuristic filename extraction.

## Recommended workflow: review, then apply

Set your source and destination paths. They must not overlap; the organizer refuses to scan when one is inside the other.

```bash
SOURCE="/path/to/downloads"
TARGET="/path/to/Media"
```

### 1. Generate a plan without changing files

```bash
python rename.py "$SOURCE" "$TARGET" \
  --generate-csv-only \
  --csv-path review.csv \
  --base-url "http://localhost:1234/v1" \
  --model "your-loaded-model"
```

This scans the source and saves a plan CSV. It does not copy, move, or clean up files. The default CSV name, if `--csv-path` is omitted, is `media_organization_plan_<timestamp>.csv` in the current directory.

### 2. Review the plan

Open `review.csv` in a spreadsheet editor. Check the title, language, season and episode details, status, and especially `Target File Path`. You can edit the target path to choose a different destination, but keep it inside the `TARGET` directory. Keep the `Source Path` values pointing to existing files under `SOURCE`.

Use statuses such as `NEEDS_REVIEW_NO_EPISODE_NUMBER`, `NEEDS_REVIEW_NO_TITLE`, and destination conflicts to find rows that need attention. Edit their target paths, or remove rows you do not want to process. When loading a reviewed CSV, the script recalculates statuses, so a review status in the CSV does not by itself prevent that row from being applied. Existing destination files are not overwritten.

### 3. Preview the reviewed plan

```bash
python rename.py "$SOURCE" "$TARGET" \
  --apply-csv review.csv \
  --action copy \
  --dry-run
```

The preview lists the planned operations without changing files. `--apply-csv` does not need a running language model because it uses the reviewed destinations from the CSV.

### 4. Apply it

```bash
python rename.py "$SOURCE" "$TARGET" \
  --apply-csv review.csv \
  --action copy
```

`--action copy` leaves the source video and subtitle files in place. Use `--action move` to move them instead. The default action is `move`.

To remove source files after a copy, add `--cleanup`. This deletes a planned source file only after all of its destination files exist. Without that option, applying a reviewed CSV does not clean up the source.

## Other useful commands

Preview a freshly generated plan directly. This writes the plan CSV to the current directory and displays the actions, without modifying files:

```bash
python rename.py "$SOURCE" "$TARGET" --action copy --dry-run
```

Run the organizer directly without reviewing a CSV first:

```bash
python rename.py "$SOURCE" "$TARGET" --action copy
```

For a direct run, the script performs source cleanup after processing: it removes files that are not recognized video or subtitle files, removes sample video clips, and removes empty folders. A dry run previews this cleanup. `--generate-csv-only` skips it. For more control, use the review-and-apply workflow above.

## Command-line options

Run `python rename.py --help` for the full help text. Common options:

| Option | Purpose |
| --- | --- |
| `SOURCE TARGET` | Source folder and organized library folder. You can also use `--source` and `--target`. |
| `--dry-run` | Show planned file operations and cleanup without changing files. |
| `--generate-csv-only` | Save the plan and exit without changing or cleaning files. |
| `--csv-path PATH` | Choose where to write the generated plan CSV. |
| `--apply-csv PATH` | Apply destinations from a reviewed plan CSV. |
| `--action copy` or `--action move` | Choose whether to copy or move; defaults to `move`. |
| `--cleanup` | Remove planned sources only after all their target files exist. |
| `--base-url URL` | OpenAI-compatible model endpoint; defaults to the configured LM Studio URL. |
| `--model NAME` | Model identifier sent to the endpoint. |
| `--api-key KEY` | API key for the model endpoint; defaults to `NA`. |
| `-j N`, `--parallelism N` | Number of concurrent planning and file-operation workers; defaults to `1`. |
| `--log-level LEVEL` | Set verbosity to `DEBUG`, `INFO`, `WARNING`, or `ERROR`. |

## Jupyter notebook and MLflow tracing

`rename.ipynb` contains a separate, interactive workflow. Open it with:

```bash
jupyter notebook rename.ipynb
```

Edit the notebook's `base_path`, `base_url`, and `model` values for your machine and model server before running its cells. The notebook enables OpenAI SDK autologging through MLflow in its first cell; the command-line script does not use MLflow tracing.

The notebook reads `MLFLOW_TRACKING_URI` and `MLFLOW_EXPERIMENT_NAME` from the environment. Their defaults are `https://mlflow.sampadhegde.in` and `torrent-naming`. Set them before launching Jupyter if you use a different MLflow server or experiment. If MLflow tracking setup fails after the package has imported, the notebook prints the error and continues.

The notebook has cells that rename files and folders. Review the paths and the cell's intended operation before running those cells; do not run all cells against your media library without checking them first.
