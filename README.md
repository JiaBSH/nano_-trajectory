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

When `analysis.compute_boundary_distances_enabled` is `true`, the analysis also
exports three per-frame distance tables. Distances are measured between closed
segmentation boundaries in nm; touching, intersecting, and containment all
produce a distance of zero:

- `nanocluster_to_nanocluster_boundary_distances.csv`: every unique particle pair.
- `nanocluster_to_nanodroplet_boundary_distances.csv`: every particle-droplet pair.
- `nanocluster_nearest_boundary_distances.csv`: the nearest particle and nearest
  droplet for each particle.

The category names are configurable through `analysis.particle_category` and
`analysis.droplet_category`; output filenames follow those values.

Set `plots.save_boundary_distance_plots` to `true` to also generate:

- `nanocluster_to_nanocluster_boundary_distance_vs_frame.png`
- `nanocluster_to_nanodroplet_boundary_distance_vs_frame.png`

Each plot uses frame id on the x-axis and boundary distance in nm on the y-axis.
Particles and droplets are tracked independently between consecutive frames using
the shared centroid-linking threshold in `output.export_max_dist_nm`. The same
value must be used in `plots.max_dist_nm`, ensuring every stable ID is identical
across CSV files, plots, and raw-frame annotations. Every stable pair ID gets
its own distance curve, and missing frames break rather than interpolate the line.
For a droplet target, `D1` is written as `instance_id=D1` in the centroid,
area, contour, diameter/height, and speed CSV files; particle targets use `P1`.
The exported value therefore matches the raw-frame annotation exactly and can be
used as a direct join/search key without adding or removing a prefix.
One-frame tracks have no computable speed and are omitted from the speed output,
but the remaining IDs are never renumbered.

When `annotations.save_boundary_pair_id_raw_frames` is `true`, a second raw-frame
overlay set is written to `annotated_boundary_pair_ids_rawframe`. It preserves the
normal all-category masks and outlines but labels particles as `P1`, `P2`, ... and
droplets as `D1`, `D2`, ... so the objects correspond directly to the pair names
in the distance-plot legend. The existing `annotated_allcat_rawframe` output is
still generated independently and is unchanged.

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
