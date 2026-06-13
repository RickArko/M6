"""Visualisation tooling — animated pipeline GIF.

Single entrypoint: :func:`render_pipeline_viz` reads scoring outputs and
emits an animated GIF showing model comparison and CV progression.
"""

from m6.viz.pipeline import render_gif, render_pipeline_viz

__all__ = [
    "render_gif",
    "render_pipeline_viz",
]
