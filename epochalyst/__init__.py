"""The epochalyst package.

It consists of the following modules:

"""

from .ensemble import EnsemblePipeline
from .model import ModelPipeline

__all__ = [
    "ModelPipeline",
    "EnsemblePipeline",
]
