"""Upstream story lab v2 - transplant workspace package.

Not imported by production. Prototypes the multi-source story architecture
(banks/packs/pipelines/styles as JSON; Python validates, routes, executes,
fails loudly) and emits the bridge artifact production will consume at the
explicit transplant chunk.
"""

from .contracts import (
    BridgeArtifact,
    LedgerWritingSpec,
    MetaMirrors,
    PipelineSpec,
    PublicDomainSourceManifest,
    Resolution,
    SourceBankSpec,
    SourceMaterialPacket,
    StoryInputPacket,
    StoryPack,
    StoryPromptProfile,
    VisualStylePolicy,
)
from .registry import Registry, RegistryError, UnknownIdError

__all__ = [
    "BridgeArtifact",
    "LedgerWritingSpec",
    "MetaMirrors",
    "PipelineSpec",
    "PublicDomainSourceManifest",
    "Registry",
    "RegistryError",
    "Resolution",
    "SourceBankSpec",
    "SourceMaterialPacket",
    "StoryInputPacket",
    "StoryPack",
    "StoryPromptProfile",
    "UnknownIdError",
    "VisualStylePolicy",
]
