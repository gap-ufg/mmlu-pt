import shutil
from collections import defaultdict
from pathlib import Path

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.text.filters import Filter, ScoreFilter
from nemo_curator.stages.text.io.reader.base import BaseReader
from nemo_curator.stages.text.io.reader.jsonl import JsonlReaderStage
from nemo_curator.stages.text.io.writer.base import BaseWriter
from nemo_curator.stages.text.modifiers import Modify
from nemo_curator.tasks import DocumentBatch, FileGroupTask, Task
from tqdm import tqdm

from mmlu_pt.utils.csv import source_to_jsonl


def prepare_sources_for_processing(original_dir, clean, manifest):
    prepare_original_dir(original_dir, clean=clean)
    for source in tqdm(manifest.sources, desc="Processing sources", unit="source"):
        source.path = Path(manifest.source_root) / source.path
        print(f"  Path: {source.path}")
        print(f"  Exam: {source.exam}")
        print(f"  Academic Level: {source.academic_level}")

        source.path = str(
            source_to_jsonl(
                Path(source.path),
                original_dir,
                exam=source.exam,
                academic_level=source.academic_level,
            )
        )

def prepare_original_dir(original_dir: Path, clean: bool) -> None:
    """Prepara o diretório de fontes, removendo seu conteúdo quando solicitado."""
    if clean and original_dir.exists():
        shutil.rmtree(original_dir)
    original_dir.mkdir(parents=True, exist_ok=True)

def get_stage_record_counts(
    pipeline: Pipeline,
    results: list[Task] | None,
) -> list[tuple[str, int, int, int]]:
    """Calcula quantos registros entraram, saíram e foram removidos por estágio."""
    stage_occurrences: defaultdict[str, int] = defaultdict(int)
    keyed_stages: list[tuple[tuple[str, int], object]] = []
    for stage in pipeline.stages:
        occurrence = stage_occurrences[stage.name]
        stage_occurrences[stage.name] += 1
        if not isinstance(stage, FilePartitioningStage):
            keyed_stages.append(((stage.name, occurrence), stage))

    input_counts: defaultdict[tuple[str, int], int] = defaultdict(int)
    for task in results or []:
        perf_occurrences: defaultdict[str, int] = defaultdict(int)
        for perf in task._stage_perf:
            occurrence = perf_occurrences[perf.stage_name]
            perf_occurrences[perf.stage_name] += 1
            input_counts[(perf.stage_name, occurrence)] += perf.num_items_processed

    inputs = [input_counts[key] for key, _ in keyed_stages]
    counts: list[tuple[str, int, int, int]] = []
    for index, (_, stage) in enumerate(keyed_stages):
        entered = inputs[index]
        exited = inputs[index + 1] if index + 1 < len(inputs) else entered
        if isinstance(stage, BaseReader):
            entered = exited
        elif isinstance(stage, BaseWriter):
            exited = entered
        counts.append((_stage_label(stage), entered, exited, max(entered - exited, 0)))
    return counts


def _stage_label(stage: object) -> str:
    """Cria um rótulo informativo a partir do tipo e do callable do estágio."""
    if isinstance(stage, Filter):
        names = [_callable_name(filter_fn) for filter_fn in stage.filter_fn]
        return f"Filter[{', '.join(names)}]"
    if isinstance(stage, ScoreFilter):
        names = [_callable_name(filter_obj) for filter_obj in stage.filter_obj]
        return f"ScoreFilter[{', '.join(names)}]"
    if isinstance(stage, Modify):
        names = [_callable_name(modifier_fn) for modifier_fn in stage.modifier_fn]
        return f"Modify[{', '.join(names)}]"
    return type(stage).__name__


def _callable_name(value: object) -> str:
    return str(getattr(value, "name", getattr(value, "__name__", type(value).__name__)))


def print_stage_record_counts(pipeline: Pipeline, results: list[Task] | None) -> None:
    """Exibe as remoções por estágio e o resumo global do pipeline."""
    counts = get_stage_record_counts(pipeline, results)
    stage_width = max(30, *(len(stage_name) for stage_name, *_ in counts))
    print(f"\nPipeline: {pipeline.name}")
    print(
        f"{'Stage':<{stage_width}} | {'Entraram':>10} | {'Saíram':>10} | "
        f"{'Removidos':>10} | {'Removidos (%)':>13}"
    )
    print(
        f"{'-' * stage_width}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}"
        f"-+-{'-' * 13}"
    )
    for stage_name, entered, exited, removed in counts:
        removed_percentage = removed / entered * 100 if entered else 0.0
        print(
            f"{stage_name:<{stage_width}} | {entered:>10} | {exited:>10} | "
            f"{removed:>10} | {removed_percentage:>12.2f}%"
        )

    original = counts[0][1] if counts else 0
    valid = counts[-1][2] if counts else 0
    invalid = max(original - valid, 0)
    valid_percentage = valid / original * 100 if original else 0.0
    invalid_percentage = invalid / original * 100 if original else 0.0
    print(
        f"\nResumo: {original} registros originais; "
        f"{valid} válidos ({valid_percentage:.2f}%); "
        f"{invalid} inválidos ({invalid_percentage:.2f}%)."
    )

class IntermediateJsonlReader(JsonlReaderStage):
    """Relê a saída intermediária preservando as métricas anteriores."""

    def process(self, task: FileGroupTask) -> DocumentBatch:
        batch = super().process(task)
        batch._stage_perf = task._stage_perf
        return batch
