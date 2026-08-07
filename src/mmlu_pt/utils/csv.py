"""Utilitários para conversão de arquivos CSV."""

from pathlib import Path

import pandas as pd


def _write_jsonl(data: pd.DataFrame, source_file: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_file = output_dir / f"{source_file.stem}.jsonl"
    data.to_json(jsonl_file, orient="records", lines=True, force_ascii=False)
    return jsonl_file


def csv_to_jsonl(csv_file: Path, output_dir: Path | None = None) -> Path:
    """Converte um CSV para JSONL e retorna o caminho gerado."""
    data = pd.read_csv(csv_file, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    return _write_jsonl(data, csv_file, output_dir or csv_file.parent)


def source_to_jsonl(
    source_file: Path,
    output_dir: Path,
    *,
    exam: str,
    academic_level: str,
) -> Path:
    """Adiciona os metadados da fonte e a salva como JSONL."""
    suffix = source_file.suffix.lower()
    if suffix == ".csv":
        data = pd.read_csv(source_file, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    elif suffix == ".jsonl":
        data = pd.read_json(source_file, lines=True, convert_dates=False)
    else:
        raise ValueError(f"formato de fonte não suportado: {suffix}")

    data["exam"] = exam
    data["academic_level"] = academic_level
    return _write_jsonl(data, source_file, output_dir)
