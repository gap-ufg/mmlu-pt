import os
import shutil

# Evita que o Ray recrie o ambiente gerenciado pelo uv em /tmp para cada execução.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

import argparse  # noqa: I001
from pathlib import Path

from nemo_curator.core.client import RayClient

from mmlu_pt.pipelines.definition import (
    DEDUPLICATION_WORK_DIR,
    ORIGINAL_DIR,
    create_deduplication_input_pipeline,
    create_duplicate_removal_workflow,
    create_exact_deduplication_workflow,
    create_pipeline,
)
from mmlu_pt.utils.manifest import read_manifest_file
from mmlu_pt.utils.pipeline_utils import (
    get_stage_record_counts,
    prepare_sources_for_processing,
    print_duplicate_removal_summary,
    print_exact_deduplication_summary,
    print_stage_record_counts,
)

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
        default=False,
        help="Limpa ORIGINAL_DIR antes de preparar as fontes (padrão: desabilitado).",
    )
    return parser


def main(manifest_file: Path = DEFAULT_MANIFEST_FILE, clean_original_dir: bool = False) -> int:
    manifest_file = manifest_file.resolve()
    manifest = read_manifest_file(manifest_file)

    prepare_sources_for_processing(ORIGINAL_DIR, clean_original_dir, manifest)
    if DEDUPLICATION_WORK_DIR.exists():
        shutil.rmtree(DEDUPLICATION_WORK_DIR)

    with RayClient(include_dashboard=False):
        pipeline = create_pipeline()
        pipeline_results = pipeline.run()

        deduplication_input_pipeline = create_deduplication_input_pipeline()
        deduplication_input_results = deduplication_input_pipeline.run()

        exact_deduplication_workflow = create_exact_deduplication_workflow()
        exact_deduplication_results = exact_deduplication_workflow.run()

        duplicate_removal_workflow = create_duplicate_removal_workflow()
        duplicate_removal_results = duplicate_removal_workflow.run()

    deduplication_input_counts = get_stage_record_counts(
        deduplication_input_pipeline,
        deduplication_input_results,
    )
    deduplication_input_records = (
        deduplication_input_counts[-1][2] if deduplication_input_counts else 0
    )

    print_stage_record_counts(pipeline, pipeline_results)
    print_stage_record_counts(
        deduplication_input_pipeline,
        deduplication_input_results,
    )
    print_exact_deduplication_summary(
        exact_deduplication_results,
        deduplication_input_records,
    )
    print_duplicate_removal_summary(
        duplicate_removal_results,
        deduplication_input_records,
    )

    print(
        "Pipeline completed successfully! "
        f"Processed {len(pipeline_results) if pipeline_results else 0} tasks."
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
