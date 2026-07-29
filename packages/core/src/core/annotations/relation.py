"""Contains the Relation class."""

from core.annotations import AnnotationId
from pydantic import BaseModel, ConfigDict


class Relation(BaseModel):
    """A directed edge from one annotation to another.

    Attributes:
        from_annotation (AnnotationId):
            The annotation this relation points away from.
        to_annotation (AnnotationId):
            The annotation this relation points towards.
        relation_type (str | None):
            The kind of relation this is. Defaults to None.
    """

    model_config = ConfigDict(frozen=True)

    from_annotation: AnnotationId
    to_annotation: AnnotationId
    relation_type: str | None = None
