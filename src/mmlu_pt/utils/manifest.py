"""Leitura e validação do manifesto de fontes."""

from pathlib import Path
from typing import Annotated, Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, StringConstraints


NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Source(BaseModel):
    """Uma fonte CSV declarada no manifesto."""

    model_config = ConfigDict(extra="forbid", strict=True)

    path: NonEmptyString
    exam: NonEmptyString
    academic_level: Literal["high_school", "undergraduate"]


class Manifest(BaseModel):
    """Schema do arquivo de manifesto."""

    model_config = ConfigDict(extra="forbid", strict=True)

    source_root: NonEmptyString
    sources: list[Source] = Field(min_length=1)


def read_manifest_file(manifest_file: Path) -> Manifest:
    """Lê um manifesto YAML e valida seu conteúdo com Pydantic."""
    manifest_data = OmegaConf.to_container(
        OmegaConf.load(manifest_file),
        resolve=True,
    )
    return Manifest.model_validate(manifest_data)
