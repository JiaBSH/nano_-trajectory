import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from rawframe_analysis.config import (
    AnalysisConfig,
    AnnotationConfig,
    ConfigError,
    InputConfig,
    JsonConfigRepository,
    OutputConfig,
    PlotConfig,
    RawFrameConfig,
)
from rawframe_analysis.pipeline import AnalysisPipeline
from rawframe_analysis.tracker import GasTracker


class ConfigTests(unittest.TestCase):
    def test_relative_paths_are_resolved_from_config_location(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            config_path = base / "configs" / "experiment.json"
            config_path.parent.mkdir()
            config_path.write_text(
                json.dumps(
                    {
                        "input": {"json_dir": "../annotations"},
                        "output": {"root": "../results/run-1"},
                    }
                ),
                encoding="utf-8",
            )

            config = (
                JsonConfigRepository()
                .load(config_path)
                .resolved_relative_to(config_path)
            )

            self.assertEqual(
                config.input.json_dir, str((base / "annotations").resolve())
            )
            self.assertEqual(
                config.output.root, str((base / "results" / "run-1").resolve())
            )

    def test_unknown_option_is_rejected(self):
        with self.assertRaisesRegex(ConfigError, "Unknown option"):
            RawFrameConfig.from_dict(
                {"input": {"json_dir": "annotations", "strict_scale_macth": True}}
            )

    def test_invalid_plot_value_is_rejected(self):
        with self.assertRaisesRegex(ConfigError, "velocity_bin_size_frames"):
            RawFrameConfig.from_dict(
                {
                    "input": {"json_dir": "annotations"},
                    "plots": {"velocity_bin_size_frames": 0},
                }
            )

    def test_pin_centroid_smoothing_alpha_must_be_in_unit_interval(self):
        for alpha in (0, -0.1, 1.1):
            with self.subTest(alpha=alpha), self.assertRaisesRegex(
                ConfigError, "pin_centroid_smoothing_alpha"
            ):
                RawFrameConfig.from_dict(
                    {
                        "input": {"json_dir": "annotations"},
                        "analysis": {"pin_centroid_smoothing_alpha": alpha},
                    }
                )

    def test_pin_centroid_smoothing_is_passed_to_tracker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            json_dir = base / "annotations"
            json_dir.mkdir()
            config = RawFrameConfig(
                input=InputConfig(json_dir=str(json_dir)),
                analysis=AnalysisConfig(pin_centroid_smoothing_alpha=0.4),
                output=OutputConfig(root=str(base / "output")),
                plots=PlotConfig(
                    save_evolution=False,
                    save_centroid_trajectories=False,
                    save_area_trajectories=False,
                    save_frame_count_area=False,
                    save_area_delta=False,
                    save_velocity_trajectories=False,
                ),
            )
            created = []

            def factory(**kwargs):
                tracker = FakeTracker(**kwargs)
                created.append(tracker)
                return tracker

            AnalysisPipeline(config, factory, log=lambda _message: None).run()

            self.assertEqual(
                created[0].constructor_kwargs["pin_centroid_smoothing_alpha"], 0.4
            )

    def test_tracking_distance_must_match_across_outputs(self):
        with self.assertRaisesRegex(ConfigError, "object IDs stay consistent"):
            RawFrameConfig.from_dict(
                {
                    "input": {"json_dir": "annotations"},
                    "output": {"export_max_dist_nm": 50.0},
                    "plots": {"max_dist_nm": 20.0},
                }
            )

    def test_category_drives_default_output_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "experiment.json"
            for category in ("gas", "nanocluster", "nanodroplet"):
                with self.subTest(category=category):
                    config = RawFrameConfig(
                        input=InputConfig(json_dir="annotations"),
                        analysis=AnalysisConfig(
                            target_category=category,
                            pin_reference_enabled=True,
                        ),
                    ).resolved_relative_to(config_path)

                    self.assertEqual(
                        Path(config.output.root).name,
                        f"{category}_pin_relative",
                    )
                    self.assertEqual(
                        config.annotations.target_output_dir,
                        f"annotated_{category}_rawframe",
                    )

    def test_version_1_gas_options_are_migrated(self):
        config = RawFrameConfig.from_dict(
            {
                "config_version": 1,
                "input": {"json_dir": "annotations"},
                "analysis": {
                    "gas_category": "nanocluster",
                    "pin_reference_enabled": True,
                },
                "output": {"root": "./nanocluster_pin_relative"},
                "annotations": {
                    "save_gas_raw_frames": True,
                    "gas_output_dir": "annotated_nanocluster_rawframe",
                    "gas_label_ids": False,
                },
            }
        )

        self.assertEqual(config.config_version, 2)
        self.assertEqual(config.analysis.target_category, "nanocluster")
        self.assertIsNone(config.output.root)
        self.assertTrue(config.annotations.save_target_raw_frames)
        self.assertIsNone(config.annotations.target_output_dir)
        self.assertFalse(config.annotations.target_label_ids)

    def test_legacy_cross_category_threshold_is_ignored(self):
        config = RawFrameConfig.from_dict(
            {
                "config_version": 2,
                "input": {"json_dir": "annotations"},
                "analysis": {"particle_in_droplet_threshold": 0.5},
            }
        )

        self.assertNotIn(
            "particle_in_droplet_threshold", config.to_dict()["analysis"]
        )


class FakeTracker:
    def __init__(self, **kwargs):
        self.constructor_kwargs = kwargs
        self.calls = []

    def process_all_frames(self):
        self.calls.append(("process_all_frames", {}))

    def export_results(self, **kwargs):
        self.calls.append(("export_results", kwargs))

    def validate_exported_csv_ids(self):
        self.calls.append(("validate_exported_csv_ids", {}))


class PipelineTests(unittest.TestCase):
    def test_pipeline_saves_effective_config_and_runs_selected_steps(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            json_dir = base / "annotations"
            json_dir.mkdir()
            output_dir = base / "output"
            config = RawFrameConfig(
                input=InputConfig(json_dir=str(json_dir)),
                output=OutputConfig(root=str(output_dir)),
                plots=PlotConfig(
                    save_evolution=False,
                    save_centroid_trajectories=False,
                    save_area_trajectories=False,
                    save_frame_count_area=False,
                    save_area_delta=False,
                    save_velocity_trajectories=False,
                ),
            )
            created = []

            def factory(**kwargs):
                tracker = FakeTracker(**kwargs)
                created.append(tracker)
                return tracker

            result = AnalysisPipeline(config, factory, log=lambda _message: None).run()

            self.assertEqual(
                result.completed_steps,
                (
                    "process_all_frames",
                    "export_results",
                    "validate_exported_csv_ids",
                ),
            )
            self.assertEqual(
                [name for name, _kwargs in created[0].calls],
                [
                    "process_all_frames",
                    "export_results",
                    "validate_exported_csv_ids",
                ],
            )
            self.assertEqual(
                created[0].constructor_kwargs["output_root"], str(output_dir)
            )
            self.assertTrue(result.config_snapshot.latest.is_file())
            self.assertTrue(result.config_snapshot.archived.is_file())
            self.assertEqual(
                JsonConfigRepository().load(result.config_snapshot.latest),
                config,
            )

    def test_real_tracker_processes_and_exports_minimal_frame(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            json_dir = base / "annotations"
            json_dir.mkdir()
            (json_dir / "frame_0001.json").write_text(
                json.dumps(
                    {
                        "objects": [
                            {
                                "category": "gas",
                                "segmentation": [[0, 0], [2, 0], [2, 2], [0, 2]],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            output_dir = base / "output"
            config = RawFrameConfig(
                input=InputConfig(json_dir=str(json_dir), nm_per_px=1.0),
                output=OutputConfig(root=str(output_dir)),
                plots=PlotConfig(
                    save_evolution=False,
                    save_centroid_trajectories=False,
                    save_area_trajectories=False,
                    save_frame_count_area=False,
                    save_area_delta=False,
                    save_velocity_trajectories=False,
                ),
            )

            result = AnalysisPipeline(
                config, GasTracker, log=lambda _message: None
            ).run()

            area_csv = output_dir / "gas_area_vs_frame.csv"
            self.assertTrue(area_csv.is_file())
            self.assertIn("4.000000", area_csv.read_text(encoding="utf-8"))
            self.assertTrue(result.config_snapshot.latest.is_file())

    def test_nanocluster_category_updates_all_derived_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            json_dir = base / "annotations"
            raw_frame_dir = base / "frames"
            json_dir.mkdir()
            raw_frame_dir.mkdir()
            objects = [
                {
                    "category": "nanocluster",
                    "segmentation": [[10, 10], [18, 10], [18, 18], [10, 18]],
                },
                {
                    "category": "pin",
                    "segmentation": [[2, 2], [6, 2], [6, 6], [2, 6]],
                },
            ]
            (json_dir / "frame_0001.json").write_text(
                json.dumps({"objects": objects}), encoding="utf-8"
            )
            Image.new("RGB", (32, 32), "white").save(raw_frame_dir / "frame_0001.png")
            config = RawFrameConfig(
                input=InputConfig(
                    json_dir=str(json_dir),
                    raw_frame_dir=str(raw_frame_dir),
                    nm_per_px=1.0,
                ),
                analysis=AnalysisConfig(
                    target_category="nanocluster",
                    pin_reference_enabled=True,
                ),
                annotations=AnnotationConfig(
                    save_target_raw_frames=True,
                    frame_step=1,
                ),
                plots=PlotConfig(
                    save_evolution=False,
                    save_centroid_trajectories=False,
                    save_area_trajectories=False,
                    save_frame_count_area=False,
                    save_area_delta=False,
                    save_velocity_trajectories=False,
                ),
            ).resolved_relative_to(base / "experiment.json")

            with contextlib.redirect_stdout(io.StringIO()):
                result = AnalysisPipeline(
                    config, GasTracker, log=lambda _message: None
                ).run()

            output_dir = Path(config.output.root)
            self.assertEqual(output_dir.name, "nanocluster_pin_relative")
            self.assertTrue((output_dir / "nanocluster_area_vs_frame.csv").is_file())
            self.assertTrue(
                (
                    output_dir / "annotated_nanocluster_rawframe" / "frame_0001.png"
                ).is_file()
            )
            self.assertEqual(result.tracker.target_category, "nanocluster")
            self.assertEqual(result.tracker.gas_category, "nanocluster")

    def test_all_annotation_and_plot_modules_run_together(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            json_dir = base / "annotations"
            raw_frame_dir = base / "frames"
            output_dir = base / "output"
            json_dir.mkdir()
            raw_frame_dir.mkdir()

            for frame_id in range(3):
                stem = f"frame_{frame_id:04d}"
                x = 10 + frame_id
                objects = [
                    {
                        "category": "gas",
                        "segmentation": [
                            [x, 10],
                            [x + 8, 10],
                            [x + 8, 18],
                            [x, 18],
                        ],
                    },
                    {
                        "category": "pin",
                        "segmentation": [[2, 2], [6, 2], [6, 6], [2, 6]],
                    },
                ]
                (json_dir / f"{stem}.json").write_text(
                    json.dumps({"objects": objects}), encoding="utf-8"
                )
                Image.new("RGB", (64, 64), "white").save(raw_frame_dir / f"{stem}.png")

            config = RawFrameConfig(
                input=InputConfig(
                    json_dir=str(json_dir),
                    raw_frame_dir=str(raw_frame_dir),
                    nm_per_px=1.0,
                ),
                output=OutputConfig(root=str(output_dir)),
                annotations=AnnotationConfig(
                    save_target_raw_frames=True,
                    save_all_category_raw_frames=True,
                    frame_step=1,
                ),
                plots=PlotConfig(
                    debug_stats=False,
                    max_tracks=10,
                    max_legend_items=10,
                    annotate_ids_max=10,
                    evolution_step=1,
                ),
            )

            with contextlib.redirect_stdout(io.StringIO()):
                result = AnalysisPipeline(
                    config, GasTracker, log=lambda _message: None
                ).run()

            expected_outputs = {
                "gas_area_vs_frame.csv",
                "gas_evolution.png",
                "gas_centroid_trajectories.png",
                "gas_area_trajectories.png",
                "gas_frame_instance_count.png",
                "gas_frame_total_area.png",
                "gas_area_delta_vs_frame.png",
                "gas_velocity_trajectories.png",
                "run_config.json",
            }
            self.assertEqual(len(result.completed_steps), 11)
            self.assertEqual(
                result.completed_steps[-1], "validate_exported_csv_ids"
            )
            self.assertTrue(
                all((output_dir / filename).is_file() for filename in expected_outputs)
            )
            self.assertEqual(
                len(list((output_dir / "annotated_gas_rawframe").glob("*.png"))),
                3,
            )
            self.assertEqual(
                len(list((output_dir / "annotated_allcat_rawframe").glob("*.png"))),
                3,
            )


if __name__ == "__main__":
    unittest.main()
