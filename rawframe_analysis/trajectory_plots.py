"""Trajectory plotting capabilities composed by plot purpose."""

from .area_trajectory_plots import AreaTrajectoryPlotMixin
from .centroid_trajectory_plots import CentroidTrajectoryPlotMixin
from .velocity_trajectory_plots import VelocityTrajectoryPlotMixin


class TrajectoryPlotMixin(
    AreaTrajectoryPlotMixin,
    VelocityTrajectoryPlotMixin,
    CentroidTrajectoryPlotMixin,
):
    """Expose the backward-compatible trajectory plotting API."""
