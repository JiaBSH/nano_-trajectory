import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager as fm
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image


CATEGORIES = ("gas", "nanocluster", "nanodroplet")
CATEGORY_LABELS = {
    "gas": "气体",
    "nanocluster": "纳米团簇",
    "nanodroplet": "纳米液滴",
}

LABEL_SIZE = 26
TICK_SIZE = 20
TITLE_SIZE = 24
ANNOTATION_SIZE = 14
CBAR_LABEL_SIZE = 24
CBAR_TICK_SIZE = 18
LEGEND_FONT_SMALL = 16
LEGEND_FONT_MEDIUM = 14
LEGEND_FONT_LARGE_COUNT = 12
EVOLUTION_LINE_WIDTH = 3.0
EVOLUTION_STEP = 50
SPATIAL_PADDING_FRACTION = 0.12
SPATIAL_MIN_PADDING_NM = 10.0
SPINE_HORIZONTAL_INSET_PX = 3


@dataclass
class ContourRecord:
    instance_id: int
    frame_id: int
    frame_name: str
    points: np.ndarray


def configure_matplotlib():
    yahei_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    chinese_font = "Microsoft YaHei"
    if yahei_path.exists():
        fm.fontManager.addfont(str(yahei_path))
        chinese_font = fm.FontProperties(fname=str(yahei_path)).get_name()

    sans_serif = [
        name
        for name in (
            chinese_font,
            "Microsoft YaHei",
            "Arial",
        )
        if name
    ]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": sans_serif,
            "font.size": TICK_SIZE,
            "axes.labelsize": LABEL_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "axes.unicode_minus": False,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LEGEND_FONT_MEDIUM,
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "axes.linewidth": 1.6,
            "xtick.major.width": 1.4,
            "ytick.major.width": 1.4,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
        }
    )


def short_title(category, plot_name):
    return f"{CATEGORY_LABELS.get(category, category)}{plot_name}"


def display_id_mapping(series_by_id):
    first_frame_by_id = {}
    for instance_id, frame_values in series_by_id.items():
        if len(frame_values) == 0:
            continue
        first_frame_by_id[int(instance_id)] = int(np.min(frame_values))
    ordered = sorted(first_frame_by_id, key=lambda iid: (first_frame_by_id[iid], iid))
    return {iid: idx + 1 for idx, iid in enumerate(ordered)}


def finish_axes(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE, labelpad=12)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE, labelpad=12)
    if title is not None:
        ax.set_title(title, fontsize=TITLE_SIZE, pad=16, loc="center")
    ax.tick_params(axis="both", labelsize=TICK_SIZE, width=1.4, length=6)
    ax.grid(True, alpha=0.25)


def save_figure(fig, out_path, extra_artists=None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"dpi": 300, "bbox_inches": "tight"}
    if extra_artists:
        kwargs["bbox_extra_artists"] = tuple(extra_artists)
    fig.savefig(out_path, **kwargs)
    plt.close(fig)


def add_track_legend(fig, ax, handles, labels):
    if not handles:
        return None

    n_items = len(handles)
    if n_items <= 20:
        fig.set_size_inches(14, 7, forward=True)
        fig.subplots_adjust(right=0.82)
        return ax.legend(
            handles=handles,
            labels=labels,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=True,
            framealpha=0.9,
            facecolor="white",
            edgecolor="gray",
            fontsize=LEGEND_FONT_SMALL,
            ncol=1,
            handlelength=1.4,
        )

    if n_items <= 60:
        fig.set_size_inches(17, 8, forward=True)
        fig.subplots_adjust(right=0.76)
        return ax.legend(
            handles=handles,
            labels=labels,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=True,
            framealpha=0.9,
            facecolor="white",
            edgecolor="gray",
            fontsize=LEGEND_FONT_MEDIUM,
            ncol=2,
            columnspacing=0.8,
            handlelength=1.2,
        )

    rows_target = 12
    ncol = int(np.ceil(float(n_items) / float(rows_target)))
    ncol = max(4, min(10, ncol))
    fig.set_size_inches(18, 12, forward=True)
    fig.subplots_adjust(bottom=0.36)
    return ax.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        frameon=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="gray",
        fontsize=LEGEND_FONT_LARGE_COUNT,
        ncol=ncol,
        columnspacing=0.8,
        handlelength=1.2,
    )


def plot_track_csv(csv_path, out_path, category, y_column, ylabel, title_kind):
    df = pd.read_csv(csv_path)
    if df.empty:
        return False

    df = df.sort_values(["instance_id", "frame_id"])
    series_frames = {
        int(instance_id): group["frame_id"].to_numpy(dtype=np.int32)
        for instance_id, group in df.groupby("instance_id", sort=True)
    }
    display_ids = display_id_mapping(series_frames)

    fig, ax = plt.subplots(figsize=(14, 7))
    cmap = plt.cm.tab20
    handles = []
    labels = []

    for instance_id, group in df.groupby("instance_id", sort=True):
        group = group.sort_values("frame_id")
        frames = group["frame_id"].to_numpy(dtype=np.int32)
        values = group[y_column].to_numpy(dtype=np.float64)
        if frames.size == 0:
            continue

        display_id = display_ids.get(int(instance_id), int(instance_id))
        color = cmap(int(display_id) % 20)
        (line,) = ax.plot(frames, values, color=color, linewidth=1.8, alpha=0.9)
        handles.append(line)
        labels.append(str(display_id))

        ax.annotate(
            str(display_id),
            xy=(float(frames[0]), float(values[0])),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=ANNOTATION_SIZE,
            color=color,
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.7,
            },
        )

    title = short_title(category, title_kind)
    finish_axes(ax, "Frame id", ylabel, title)
    legend = add_track_legend(fig, ax, handles, labels)
    fig.tight_layout()
    save_figure(fig, out_path, extra_artists=[legend] if legend is not None else None)
    return True


def plot_area_delta(csv_path, out_path, category):
    df = pd.read_csv(csv_path)
    if df.empty:
        return False

    y_col = "delta_area_nm2_per_frame"
    if y_col not in df.columns:
        y_col = [col for col in df.columns if col.startswith("delta_area")][0]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df["frame_id"], df[y_col], color="#1f77b4", linewidth=2.2)
    ax.axhline(0.0, color="black", linewidth=1.4, alpha=0.65)
    finish_axes(
        ax,
        "Frame id",
        "Delta Area (nm^2/frame)",
        short_title(category, "总面积变化"),
    )
    fig.tight_layout()
    save_figure(fig, out_path)
    return True


def read_contours(csv_path):
    records = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) < 4:
                continue
            points = []
            for item in row[3:]:
                item = item.strip()
                if not item:
                    continue
                if item.startswith("(") and item.endswith(")"):
                    item = item[1:-1]
                try:
                    x_text, y_text = item.split(",", 1)
                    points.append((float(x_text), float(y_text)))
                except ValueError:
                    continue
            if len(points) == 0:
                continue
            records.append(
                ContourRecord(
                    instance_id=int(row[0]),
                    frame_id=int(row[1]),
                    frame_name=row[2],
                    points=np.array(points, dtype=np.float64),
                )
            )
    return records


def first_rawframe_size(category_dir, category):
    raw_dir = category_dir / f"annotated_{category}_rawframe"
    if not raw_dir.exists():
        return None
    first_png = next(iter(sorted(raw_dir.glob("*.png"))), None)
    if first_png is None:
        return None
    with Image.open(first_png) as image:
        return image.size


def max_nm_per_pixel(category_dir, category):
    for suffix in ("centroids", "area_vs_frame", "diameter_height_vs_frame"):
        csv_path = category_dir / f"{category}_{suffix}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path, usecols=["nm_per_pixel"])
        if not df.empty:
            return float(df["nm_per_pixel"].max())
    return None


def spatial_limits(category_dir, category, contours=None, centroids=None):
    xs = []
    ys = []
    if contours:
        for record in contours:
            xs.extend(record.points[:, 0])
            ys.extend(record.points[:, 1])
    if centroids is not None and not centroids.empty:
        xs.extend(centroids["cx_nm"].to_numpy(dtype=np.float64))
        ys.extend(centroids["cy_nm"].to_numpy(dtype=np.float64))
    if not xs or not ys:
        scale = max_nm_per_pixel(category_dir, category)
        raw_size = first_rawframe_size(category_dir, category)
        if scale is not None and raw_size is not None:
            width_px, height_px = raw_size
            return (0.0, float(width_px) * scale * 1.5), (float(height_px) * scale, 0.0)
        return None, None

    x_min = float(np.min(xs))
    x_max = float(np.max(xs))
    y_min = float(np.min(ys))
    y_max = float(np.max(ys))
    data_span = max(x_max - x_min, y_max - y_min)
    pad = max(data_span * SPATIAL_PADDING_FRACTION, SPATIAL_MIN_PADDING_NM)
    return (x_min - pad, x_max + pad), (y_max + pad, y_min - pad)


def shared_contour_limits(input_root, categories):
    xs = []
    ys = []
    for category in categories:
        contour_path = input_root / category / f"{category}_contours_by_frame.csv"
        if not contour_path.exists():
            continue
        for record in read_contours(contour_path):
            xs.extend(record.points[:, 0])
            ys.extend(record.points[:, 1])

    if not xs or not ys:
        return None

    x_min = float(np.min(xs))
    x_max = float(np.max(xs))
    y_min = float(np.min(ys))
    y_max = float(np.max(ys))
    data_span = max(x_max - x_min, y_max - y_min)
    pad = max(data_span * SPATIAL_PADDING_FRACTION, SPATIAL_MIN_PADDING_NM)
    return (x_min - pad, x_max + pad), (y_max + pad, y_min - pad)


def setup_spatial_axes(fig, ax, category_dir, category, contours=None, centroids=None, limits=None):
    if limits is None:
        xlim, ylim = spatial_limits(category_dir, category, contours=contours, centroids=centroids)
    else:
        xlim, ylim = limits
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (nm)", fontsize=LABEL_SIZE, labelpad=12)
    ax.set_ylabel("y (nm)", fontsize=LABEL_SIZE, labelpad=12)
    ax.tick_params(axis="both", labelsize=TICK_SIZE, width=1.4, length=6)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.14)
    return cax


def add_spatial_border(ax):
    border = Rectangle(
        (0, 0),
        1,
        1,
        transform=ax.transAxes,
        fill=False,
        edgecolor="black",
        linewidth=3.0,
        zorder=10,
        clip_on=False,
    )
    ax.add_patch(border)


def frame_bbox_from_spines(image_path):
    image = Image.open(image_path).convert("RGB")
    arr = np.asarray(image)
    black = (arr[:, :, 0] < 40) & (arr[:, :, 1] < 40) & (arr[:, :, 2] < 40)
    row_counts = black.sum(axis=1)
    rows = np.where(row_counts > arr.shape[1] * 0.25)[0]
    if rows.size == 0:
        raise ValueError(f"Could not detect horizontal axes spines: {image_path}")

    groups = []
    for row in rows:
        if not groups or row > groups[-1][-1] + 1:
            groups.append([int(row)])
        else:
            groups[-1].append(int(row))

    y_top = int(round(float(np.mean(groups[0]))))
    y_bottom = int(round(float(np.mean(groups[-1]))))
    cols_top = np.where(black[y_top])[0]
    cols_bottom = np.where(black[y_bottom])[0]
    cols = np.intersect1d(cols_top, cols_bottom)
    if cols.size == 0:
        cols = np.union1d(cols_top, cols_bottom)
    if cols.size == 0:
        raise ValueError(f"Could not detect vertical axes spines: {image_path}")

    return int(cols.min()), y_top, int(cols.max()), y_bottom


def plot_area_delta_like_reference(csv_path, out_path, category, reference_path):
    df = pd.read_csv(csv_path)
    if df.empty:
        return False

    y_col = "delta_area_nm2_per_frame"
    if y_col not in df.columns:
        y_col = [col for col in df.columns if col.startswith("delta_area")][0]

    with Image.open(reference_path) as reference_image:
        target_w, target_h = reference_image.size

    ref_x0, ref_y_top, ref_x1, ref_y_bottom = frame_bbox_from_spines(reference_path)
    rect_x0 = ref_x0 + SPINE_HORIZONTAL_INSET_PX
    rect_x1 = ref_x1 - SPINE_HORIZONTAL_INSET_PX

    dpi = 300
    fig = plt.figure(figsize=(target_w / dpi, target_h / dpi), dpi=dpi, facecolor="white")
    ax = fig.add_axes(
        [
            rect_x0 / target_w,
            (target_h - ref_y_bottom) / target_h,
            (rect_x1 - rect_x0) / target_w,
            (ref_y_bottom - ref_y_top) / target_h,
        ]
    )
    ax.plot(df["frame_id"], df[y_col], color="#1f77b4", linewidth=2.2)
    ax.axhline(0.0, color="black", linewidth=1.4, alpha=0.65)
    finish_axes(
        ax,
        "Frame id",
        "Delta Area (nm^2/frame)",
        short_title(category, "\u603b\u9762\u79ef\u53d8\u5316"),
    )
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return True


def plot_evolution(category_dir, out_path, category, step=EVOLUTION_STEP, limits=None):
    contour_path = category_dir / f"{category}_contours_by_frame.csv"
    contours = read_contours(contour_path)
    if not contours:
        return False

    max_frame = max(record.frame_id for record in contours)
    norm = Normalize(vmin=0, vmax=max_frame)
    cmap = plt.cm.plasma

    fig, ax = plt.subplots(figsize=(10, 10))
    cax = setup_spatial_axes(fig, ax, category_dir, category, contours=contours, limits=limits)

    for record in contours:
        if record.frame_id % step != 0:
            continue
        points = record.points
        if points.shape[0] < 2:
            continue
        closed = np.vstack([points, points[0]])
        ax.plot(
            closed[:, 0],
            closed[:, 1],
            color=cmap(norm(record.frame_id)),
            linewidth=EVOLUTION_LINE_WIDTH,
            alpha=0.9,
        )

    scalar = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    scalar.set_array([])
    cbar = fig.colorbar(scalar, cax=cax)
    cbar.set_label("Frame id", fontsize=CBAR_LABEL_SIZE, labelpad=14)
    cbar.ax.tick_params(labelsize=CBAR_TICK_SIZE, width=1.3, length=5)
    ax.set_title(short_title(category, "轮廓演化"), fontsize=TITLE_SIZE, pad=16)
    add_spatial_border(ax)
    fig.tight_layout()
    save_figure(fig, out_path)
    return True


def plot_centroid_trajectories(category_dir, out_path, category):
    centroid_path = category_dir / f"{category}_centroids.csv"
    df = pd.read_csv(centroid_path)
    if df.empty:
        return False

    max_frame = int(df["frame_id"].max())
    norm = Normalize(vmin=0, vmax=max_frame)
    cmap = plt.cm.plasma

    fig, ax = plt.subplots(figsize=(10, 10))
    cax = setup_spatial_axes(fig, ax, category_dir, category, centroids=df)

    for _instance_id, group in df.groupby("instance_id", sort=True):
        group = group.sort_values("frame_id")
        frames = group["frame_id"].to_numpy(dtype=np.float64)
        points = group[["cx_nm", "cy_nm"]].to_numpy(dtype=np.float64)
        if points.shape[0] == 0:
            continue
        if points.shape[0] > 1:
            segments = np.stack([points[:-1], points[1:]], axis=1)
            collection = LineCollection(
                segments,
                colors=[cmap(norm(frame)) for frame in frames[:-1]],
                linewidths=1.8,
                alpha=0.95,
            )
            ax.add_collection(collection)
        ax.scatter(points[:, 0], points[:, 1], c=frames, cmap=cmap, norm=norm, s=8)

    scalar = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    scalar.set_array([])
    cbar = fig.colorbar(scalar, cax=cax)
    cbar.set_label("Frame id", fontsize=CBAR_LABEL_SIZE, labelpad=14)
    cbar.ax.tick_params(labelsize=CBAR_TICK_SIZE, width=1.3, length=5)
    ax.set_title(short_title(category, "质心轨迹"), fontsize=TITLE_SIZE, pad=16)
    add_spatial_border(ax)
    fig.tight_layout()
    save_figure(fig, out_path)
    return True


def replot_category(input_root, output_root, category, shared_evolution_limits=None):
    category_dir = input_root / category
    output_dir = output_root / category
    made = []

    jobs = [
        (
            plot_area_delta,
            category_dir / f"{category}_area_delta_vs_frame.csv",
            output_dir / f"{category}_area_delta_vs_frame.png",
            category,
        ),
        (
            plot_track_csv,
            category_dir / f"{category}_instance_area_vs_frame.csv",
            output_dir / f"{category}_area_trajectories.png",
            category,
            "area_nm2",
            "Area (nm^2)",
            "面积变化",
        ),
        (
            plot_track_csv,
            category_dir / f"{category}_instance_speed_vs_frame.csv",
            output_dir / f"{category}_velocity_trajectories.png",
            category,
            "speed_nm_per_s",
            "Speed (nm/s)",
            "速度变化",
        ),
        (
            plot_evolution,
            category_dir,
            output_dir / f"{category}_evolution.png",
            category,
            EVOLUTION_STEP,
            shared_evolution_limits,
        ),
        (
            plot_centroid_trajectories,
            category_dir,
            output_dir / f"{category}_centroid_trajectories.png",
            category,
        ),
    ]

    for job in jobs:
        func = job[0]
        args = job[1:]
        try:
            ok = func(*args)
        except Exception as exc:
            print(f"[fail] {category}: {func.__name__}: {exc}")
            ok = False
        if ok:
            made.append(args[1])

    return made


def main():
    parser = argparse.ArgumentParser(
        description="Replot result/0513 figures with larger axis labels."
    )
    parser.add_argument("--input-root", default=r"result\0513")
    parser.add_argument("--output-root", default=r"result\0513_large_labels")
    parser.add_argument("--categories", nargs="*", default=list(CATEGORIES))
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    configure_matplotlib()
    output_root.mkdir(parents=True, exist_ok=True)
    shared_evolution_limits = shared_contour_limits(input_root, args.categories)

    total = 0
    for category in args.categories:
        made = replot_category(input_root, output_root, category, shared_evolution_limits=shared_evolution_limits)
        total += len(made)
        for path in made:
            print(f"[ok] {path}")

    if "nanocluster" in args.categories:
        fixed = plot_area_delta_like_reference(
            input_root / "nanocluster" / "nanocluster_area_delta_vs_frame.csv",
            output_root / "nanocluster" / "nanocluster_area_delta_vs_frame.png",
            "nanocluster",
            output_root / "nanocluster" / "nanocluster_velocity_trajectories.png",
        )
        if fixed:
            print("[ok] matched nanocluster_area_delta_vs_frame.png to nanocluster_velocity_trajectories.png")

    print(f"Done. Generated {total} figures under {output_root.resolve()}")


if __name__ == "__main__":
    main()
