import argparse
import os
from pathlib import Path

from mmlu_pt.utils.pipeline_utils import (
    IntermediateJsonlReader,
    prepare_sources_for_processing,
    print_stage_record_counts,
)

# Evita que o Ray recrie o ambiente gerenciado pelo uv em /tmp para cada execução.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.filters import Filter
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.text.modifiers import Modify

from mmlu_pt.mcqa_minimal import (
    PUBLIC_FIELDS,
    extract_choices,
    has_answer,
    has_described_choices,
    has_matching_alternative_lengths,
    has_minimum_question_length,
    has_supported_choice_count,
    keep_question,
    normalize_answer,
    parse_alternatives,
)
from mmlu_pt.utils.manifest import read_manifest_file

OUTPUT_DIR = Path("output").resolve()
ORIGINAL_DIR = OUTPUT_DIR / "01 - original"
READ_DIR = OUTPUT_DIR / "02 - read"
FILTERED_DIR = OUTPUT_DIR / "03 - filtered"
DEFAULT_MANIFEST_FILE = Path("config/sources.yaml")


def build_parser() -> argparse.ArgumentParser:
    """Cria o parser dos parâmetros do pipeline."""
    parser = argparse.ArgumentParser(description="Executa o pipeline mínimo do MMLU-PT.")
    parser.add_argument(
        "--config",
        "--manifest-file",
        dest="manifest_file",
        type=Path,
        default=DEFAULT_MANIFEST_FILE,
        help="Caminho do manifest YAML (padrão: config/sources.yaml).",
    )
    parser.add_argument(
        "--clean-original-dir",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Limpa ORIGINAL_DIR antes de preparar as fontes (padrão: habilitado).",
    )
    return parser


def create_pipeline() -> Pipeline:
    """Cria o pipeline completo com escrita JSONL intermediária."""
    return Pipeline(
        name="mmlu_pt",
        description="Read, normalize and filter the MMLU-PT dataset",
        stages=[
            JsonlReader(file_paths=str(ORIGINAL_DIR)),
            Modify(keep_question, input_fields="statement", output_fields="question"),
            Modify(parse_alternatives, input_fields="alternatives", output_fields="parsed"),
            Modify(normalize_answer, input_fields="answer", output_fields="answer"),
            Modify(extract_choices, input_fields="parsed", output_fields="choices"),
            JsonlWriter(
                path=str(READ_DIR),
                # fields=PUBLIC_FIELDS,
                write_kwargs={"index": False},
                mode="overwrite",
            ),
            IntermediateJsonlReader(),
            Filter(
                has_matching_alternative_lengths,
                filter_field="parsed",
            ),
            Filter(
                has_described_choices,
                filter_field="choices",
            ),
            Filter(
                has_supported_choice_count,
                filter_field="choices",
            ),
            Filter(
                has_answer,
                filter_field="answer",
            ),
            Filter(
                has_minimum_question_length,
                filter_field="question",
            ),
            JsonlWriter(
                path=str(FILTERED_DIR),
                fields=PUBLIC_FIELDS,
                write_kwargs={"index": False},
                mode="overwrite",
            ),
        ],
    )

def main(manifest_file: Path = DEFAULT_MANIFEST_FILE, clean_original_dir: bool = True) -> int:
    manifest_file = manifest_file.resolve()
    manifest = read_manifest_file(manifest_file)

    prepare_sources_for_processing(ORIGINAL_DIR, clean_original_dir, manifest)

    with RayClient(include_dashboard=False):
        pipeline = create_pipeline()
        results = pipeline.run()
        print_stage_record_counts(pipeline, results)

    print(
        "Pipeline completed successfully! "
        f"Processed {len(results) if results else 0} tasks."
    )
    return 0


if __name__ == "__main__":
    args = build_parser().parse_args()
    raise SystemExit(
        main(
            manifest_file=args.manifest_file,
            clean_original_dir=args.clean_original_dir,
        )
    )
