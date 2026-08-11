"""Top-level inference-release manifest contract."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from common.release.base import ReleaseModel
from common.release.config import InferenceConfig
from common.release.metrics import MetricsSpec
from common.release.preprocessor import PreprocessorSpec
from common.release.provenance import ProvenanceSpec


class ReleaseManifest(ReleaseModel):
    """Single versioned document describing an inference-ready release."""

    format_version: Literal["1.0"] = "1.0"
    release_id: str = Field(min_length=1)
    config: InferenceConfig
    preprocessor: PreprocessorSpec
    metrics: MetricsSpec
    provenance: ProvenanceSpec
