# nano_-trajectory

Extract nano trajectories from frame-by-frame segmentation annotations.

## Raw-frame analysis

All runtime parameters are stored in `analyze-rawframe.config.json`. Relative
paths in that file are resolved relative to the config file itself, so runs do
not depend on the current working directory.

The analyzed category has a single source of truth:

```json
"analysis": {
  "target_category": "nanocluster"
}
```

Keep `output.root` and `annotations.target_output_dir` set to `null`. They are
then derived automatically as `nanocluster_pin_relative` and
`annotated_nanocluster_rawframe`. Every CSV and plot filename already uses the
same category prefix. Custom names may use the `{category}` placeholder, for
example `"root": "./result/{category}"`.

```powershell
# Edit analyze-rawframe.config.json first, then run:
python analyze-rawframe.py

# Use another config file:
python analyze-rawframe.py --config path\to\experiment.json

# Check parameter values and input paths without analyzing data:
python analyze-rawframe.py --config path\to\experiment.json --validate-only
```

Every successful run writes two copies of the fully resolved configuration:

- `<output.root>/run_config.json` is the latest effective configuration.
- `<output.root>/run_configs/run_config_YYYYMMDD_HHMMSS_ffffff.json` is the
  immutable configuration snapshot for that run.

The code is separated by responsibility:

- `rawframe_analysis/config.py`: typed configuration, validation, and snapshots.
- `rawframe_analysis/pipeline.py`: configurable run orchestration.
- `rawframe_analysis/tracker.py`: lightweight state initialization and capability composition.
- `rawframe_analysis/inputs.py`, `geometry.py`, `processing.py`, `tracking.py`:
  focused analysis stages.
- `rawframe_analysis/exporting.py`: CSV serialization.
- annotation and plotting modules: one renderer or plot family per file.
- `analyze-rawframe.py`: backward-compatible command-line entry point.

See `rawframe_analysis/ARCHITECTURE.md` for the responsibility map and call flow.
