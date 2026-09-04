from pathlib import Path

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.deduplication.exact.workflow import ExactDeduplicationWorkflow
from nemo_curator.stages.text.deduplication.removal_workflow import (
    TextDuplicatesRemovalWorkflow,
)
from nemo_curator.stages.text.filters import Filter, ScoreFilter
from nemo_curator.stages.text.filters.heuristic import WordCountFilter
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.text.modifiers import Modify

from mmlu_pt.mcqa_minimal import (
    PUBLIC_FIELDS,
    extract_choices,
    has_answer,
    has_described_choices,
    has_matching_alternative_lengths,
    has_supported_choice_count,
    keep_question,
    normalize_answer,
    normalize_question_for_dedup,
    parse_alternatives,
)
from mmlu_pt.utils.pipeline_utils import IntermediateJsonlReader

OUTPUT_DIR = Path("output").resolve()
ORIGINAL_DIR = OUTPUT_DIR / "01 - original"
READ_DIR = OUTPUT_DIR / "02 - read"
PRE_WORD_FILTER_DIR = OUTPUT_DIR / "03 - pre-word-filter"
FILTERED_DIR = OUTPUT_DIR / "04 - filtered"
DEDUPLICATION_WORK_DIR = OUTPUT_DIR / "exact-deduplication-work"
DEDUPLICATION_INPUT_DIR = DEDUPLICATION_WORK_DIR / "input"
DEDUPLICATION_RESULTS_DIR = DEDUPLICATION_WORK_DIR / "results"
EXACT_DUPLICATE_IDS_DIR = DEDUPLICATION_RESULTS_DIR / "ExactDuplicateIds"
EXACT_ID_GENERATOR_PATH = DEDUPLICATION_RESULTS_DIR / "exact_id_generator.json"
DEDUPLICATED_DIR = OUTPUT_DIR / "05 - deduplicated"
NORMALIZED_QUESTION_FIELD = "question_normalized"
DEDUPLICATION_FIELDS = [*PUBLIC_FIELDS, NORMALIZED_QUESTION_FIELD]
DEDUPLICATION_INPUT_BLOCKSIZE = "256MiB"


def create_pipeline() -> Pipeline:
    """Cria o pipeline de leitura, normalização e filtragem."""
    return Pipeline(
        name="mmlu_pt",
        description="Read, normalize and filter the MMLU-PT dataset",
        stages=[
            JsonlReader(file_paths=str(ORIGINAL_DIR)),
            Modify(keep_question, input_fields="statement", output_fields="question"),
            Modify(
                parse_alternatives,
                input_fields="alternatives",
                output_fields="parsed",
            ),
            Modify(normalize_answer, input_fields="answer", output_fields="answer"),
            Modify(extract_choices, input_fields="parsed", output_fields="choices"),
            JsonlWriter(
                path=str(READ_DIR),
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
            JsonlWriter(
                path=str(PRE_WORD_FILTER_DIR),
                fields=PUBLIC_FIELDS,
                write_kwargs={"index": False},
                mode="overwrite",
            ),
            IntermediateJsonlReader(),
            ScoreFilter(
                filter_obj=WordCountFilter(
                    min_words=4,
                    max_words=1_000,
                    lang="pt",
                ),
                text_field="question",
            ),
            JsonlWriter(
                path=str(FILTERED_DIR),
                fields=PUBLIC_FIELDS,
                write_kwargs={"index": False},
                mode="overwrite",
            ),
        ],
    )


def create_deduplication_input_pipeline() -> Pipeline:
    """Cria o pipeline que materializa a pergunta normalizada."""
    return Pipeline(
        name="mmlu_pt_exact_deduplication_input",
        description="Normalize questions for exact deduplication",
        stages=[
            JsonlReader(file_paths=str(FILTERED_DIR)),
            Modify(
                normalize_question_for_dedup,
                input_fields="question",
                output_fields=NORMALIZED_QUESTION_FIELD,
            ),
            JsonlWriter(
                path=str(DEDUPLICATION_INPUT_DIR),
                fields=DEDUPLICATION_FIELDS,
                write_kwargs={"index": False},
                mode="overwrite",
            ),
        ],
    )


def create_exact_deduplication_workflow() -> ExactDeduplicationWorkflow:
    """Cria o workflow que identifica perguntas normalizadas duplicadas."""
    return ExactDeduplicationWorkflow(
        input_path=str(DEDUPLICATION_INPUT_DIR),
        output_path=str(DEDUPLICATION_RESULTS_DIR),
        text_field=NORMALIZED_QUESTION_FIELD,
        input_filetype="jsonl",
        input_blocksize=DEDUPLICATION_INPUT_BLOCKSIZE,
        perform_removal=False,
        assign_id=True,
    )


def create_duplicate_removal_workflow() -> TextDuplicatesRemovalWorkflow:
    """Cria o workflow que remove os IDs de perguntas duplicadas."""
    return TextDuplicatesRemovalWorkflow(
        input_path=str(DEDUPLICATION_INPUT_DIR),
        ids_to_remove_path=str(EXACT_DUPLICATE_IDS_DIR),
        output_path=str(DEDUPLICATED_DIR),
        input_filetype="jsonl",
        input_blocksize=DEDUPLICATION_INPUT_BLOCKSIZE,
        id_field="_curator_dedup_id",
        duplicate_id_field="_curator_dedup_id",
        id_generator_path=str(EXACT_ID_GENERATOR_PATH),
        output_filetype="jsonl",
        output_kwargs={"index": False},
        output_fields=PUBLIC_FIELDS,
        output_mode="overwrite",
    )
