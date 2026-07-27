"""Annotation capabilities composed from focused renderers."""

from .all_category_annotations import AllCategoryAnnotationMixin
from .annotated_images import AnnotatedImageMixin
from .annotation_support import AnnotationSupportMixin
from .rawframe_annotations import RawFrameAnnotationMixin


class AnnotationMixin(
    AnnotationSupportMixin,
    AnnotatedImageMixin,
    RawFrameAnnotationMixin,
    AllCategoryAnnotationMixin,
):
    """Expose the backward-compatible annotation API."""
