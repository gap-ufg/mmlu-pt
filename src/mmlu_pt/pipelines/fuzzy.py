import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from mmlu_pt.mcqa_minimal import PUBLIC_FIELDS, normalize_question_for_dedup

CHAR_NGRAMS = 8
NUM_BANDS = 10
HASHES_PER_BAND = 26
NUM_HASHES = NUM_BANDS * HASHES_PER_BAND
MIN_MATCHING_BANDS = 1
JACCARD_THRESHOLD = 0.94
SEED = 42
BATCH_SIZE = 2_048
PARAMETERS = {
    "char_ngrams": CHAR_NGRAMS,
    "num_bands": NUM_BANDS,
    "minhashes_per_band": HASHES_PER_BAND,
    "min_matching_bands": MIN_MATCHING_BANDS,
    "jaccard_threshold": JACCARD_THRESHOLD,
    "seed": SEED,
    "num_hashes": NUM_HASHES,
    "use_64_bit_hash": False,
    "representative_policy": "filled_fields_then_text_length_then_stable_id_v1",
}


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def serialize_fuzzy_text(question: str, choices: list[str]) -> str:
    """Serializa o item com o mesmo protocolo do benchmark pareado."""
    labeled = "\n".join(f"[{chr(ord('A') + index)}] {choice}" for index, choice in enumerate(choices))
    return normalize_question_for_dedup(f"{question}\n{labeled}")


def read_records(input_dir: Path) -> tuple[list[dict], list[dict]]:
    """Lê os campos públicos e identifica cada ocorrência deterministicamente."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Diretório de entrada fuzzy ausente: {input_dir}")
    records, manifest = [], []
    occurrences = Counter()
    for path in sorted(input_dir.glob("*.jsonl")):
        original_hash = file_sha256(path)
        count = 0
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    if not isinstance(raw, dict):
                        raise ValueError("O registro deve ser um objeto JSON")
                    question, choices = raw.get("question"), raw.get("choices")
                    if not isinstance(question, str):
                        raise ValueError("question deve ser uma string")
                    if not isinstance(choices, list) or not all(isinstance(choice, str) for choice in choices):
                        raise ValueError("choices deve ser uma lista de strings")
                    public = {field: raw.get(field) for field in PUBLIC_FIELDS}
                    payload = canonical_json(public)
                except (ValueError, TypeError) as error:
                    raise ValueError(f"Registro inválido em {path}:{line_number}: {error}") from error
                digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
                occurrence = occurrences[digest]
                occurrences[digest] += 1
                filled = sum(value is not None and value != "" and value != [] for value in public.values())
                records.append({
                    "id": f"{digest}:{occurrence:08d}",
                    "public": public,
                    "text": serialize_fuzzy_text(question, choices),
                    "filled_fields": filled,
                    "source": path.name,
                    "line": line_number,
                })
                count += 1
        if file_sha256(path) != original_hash:
            raise RuntimeError(f"A entrada mudou durante a leitura: {path}")
        manifest.append({"file": path.name, "sha256": original_hash, "records": count})
    records.sort(key=lambda row: (-row["filled_fields"], -len(row["text"]), row["id"]))
    return records, manifest


def compute_signatures(texts: list[str]) -> tuple[np.ndarray, dict]:
    """Calcula MinHash na GPU em lotes limitados."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    if not texts:
        return np.empty((0, NUM_HASHES), dtype=np.uint32), {"used": False}
    try:
        import cudf
        import cupy as cp
        from nemo_curator.stages.deduplication.fuzzy.minhash import GPUMinHash

        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("Nenhuma GPU CUDA visível")
        device = cp.cuda.Device()
        properties = cp.cuda.runtime.getDeviceProperties(device.id)
        name = properties["name"]
        gpu = {
            "used": True,
            "logical_device": device.id,
            "name": name.decode() if isinstance(name, bytes) else name,
            "driver_version": cp.cuda.runtime.driverGetVersion(),
            "runtime_version": cp.cuda.runtime.runtimeGetVersion(),
        }
        processor = GPUMinHash(
            seed=SEED, num_hashes=NUM_HASHES, char_ngrams=CHAR_NGRAMS,
            use_64bit_hash=False, pool=False,
        )
        signatures = np.empty((len(texts), NUM_HASHES), dtype=np.uint32)
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start:start + BATCH_SIZE]
            values = processor.compute_minhashes(cudf.Series(batch))
            signatures[start:start + len(batch)] = np.asarray(values.to_arrow().to_pylist(), dtype=np.uint32)
        return signatures, gpu
    except Exception as error:
        raise RuntimeError(
            "Falha no MinHash fuzzy: é necessária uma GPU NVIDIA/CUDA compatível "
            f"com NeMo Curator (CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}). {error}"
        ) from error


def band_keys(signature: np.ndarray) -> list[tuple]:
    return [(index, *map(int, band)) for index, band in enumerate(signature.reshape(NUM_BANDS, HASHES_PER_BAND))]


def matching_bands(left: np.ndarray, right: np.ndarray) -> int:
    return int((left == right).reshape(NUM_BANDS, HASHES_PER_BAND).all(axis=1).sum())


def jaccard(left: set[str], right: set[str]) -> float:
    intersection_size = len(left & right)
    union_size = len(left) + len(right) - intersection_size
    return intersection_size / union_size if union_size else 0.0


def select_representatives(records: list[dict], signatures: np.ndarray) -> tuple[list[int], list[dict], dict]:
    """Remove somente itens com correspondência direta a um representante mantido."""
    eligible = [index for index, record in enumerate(records) if len(record["text"]) >= CHAR_NGRAMS]
    if signatures.shape != (len(eligible), NUM_HASHES) or signatures.dtype != np.uint32:
        raise ValueError("Matriz MinHash incompatível com os registros elegíveis")
    signature_rows = {index: offset for offset, index in enumerate(eligible)}
    buckets = defaultdict(list)
    kept, removals = [], []
    comparisons = 0

    @lru_cache(maxsize=512)
    def shingles(index: int) -> set[str]:
        text = records[index]["text"]
        return {text[start:start + CHAR_NGRAMS] for start in range(len(text) - CHAR_NGRAMS + 1)}

    for index, record in enumerate(records):
        if index not in signature_rows:
            kept.append(index)
            continue
        signature = signatures[signature_rows[index]]
        keys = band_keys(signature)
        candidates = sorted({candidate for key in keys for candidate in buckets.get(key, ())})
        for candidate in candidates:
            comparisons += 1
            similarity = jaccard(shingles(index), shingles(candidate))
            if similarity < JACCARD_THRESHOLD:
                continue
            bands = matching_bands(signature, signatures[signature_rows[candidate]])
            if bands < MIN_MATCHING_BANDS:
                continue
            removals.append({
                "removed_id": record["id"],
                "representative_id": records[candidate]["id"],
                "jaccard": similarity,
                "matching_bands": bands,
            })
            break
        else:
            kept.append(index)
            for key in keys:
                buckets[key].append(index)

    kept_ids = {records[index]["id"] for index in kept}
    removed_ids = {row["removed_id"] for row in removals}
    if kept_ids & removed_ids or len(kept) + len(removals) != len(records):
        raise AssertionError("Partição fuzzy inválida")
    for removal in removals:
        if removal["representative_id"] not in kept_ids:
            raise AssertionError("O representante deve permanecer no output")
        if removal["jaccard"] < JACCARD_THRESHOLD or removal["matching_bands"] < MIN_MATCHING_BANDS:
            raise AssertionError("Remoção sem correspondência direta")
    return kept, removals, {"short_texts_preserved": len(records) - len(eligible), "candidate_comparisons": comparisons}


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(canonical_json(row) + "\n")


def run_fuzzy_deduplication(input_dir: Path, output_dir: Path, work_dir: Path) -> dict:
    """Executa a etapa fuzzy e publica o dataset somente após validá-lo."""
    started = time.perf_counter()
    input_dir, output_dir, work_dir = input_dir.resolve(), output_dir.resolve(), work_dir.resolve()
    if input_dir == output_dir or input_dir in output_dir.parents or output_dir in input_dir.parents:
        raise ValueError("As pastas de entrada e saída fuzzy devem ser separadas")
    records, manifest = read_records(input_dir)
    fingerprint = hashlib.sha256(canonical_json({"inputs": manifest, "parameters": PARAMETERS}).encode()).hexdigest()
    eligible_texts = [record["text"] for record in records if len(record["text"]) >= CHAR_NGRAMS]
    print(f"\nFuzzy: {len(records)} registros; calculando MinHash para {len(eligible_texts)} itens.", flush=True)
    signatures, gpu = compute_signatures(eligible_texts)
    kept, removals, diagnostics = select_representatives(records, signatures)

    run_parent = work_dir / fingerprint
    run_parent.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=run_parent))
    write_jsonl(run_dir / "removals.jsonl", removals)
    kept_set = set(kept)
    write_jsonl(run_dir / "records.jsonl", (
        {"id": row["id"], "source": row["source"], "line": row["line"], "kept": index in kept_set}
        for index, row in enumerate(records)
    ))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".fuzzy-output-", dir=output_dir.parent) as temporary:
        staged = Path(temporary) / "dataset"
        staged.mkdir()
        dataset_path = staged / "data.jsonl"
        write_jsonl(dataset_path, (records[index]["public"] for index in kept))
        with dataset_path.open(encoding="utf-8") as stream:
            output_count = sum(1 for line in stream if set(json.loads(line)) == set(PUBLIC_FIELDS))
        if output_count != len(kept):
            raise AssertionError("Contagem ou schema do output fuzzy inválido")
        if sorted(path.name for path in input_dir.glob("*.jsonl")) != [row["file"] for row in manifest]:
            raise RuntimeError("A lista de arquivos de entrada mudou durante a deduplicação")
        for row in manifest:
            if file_sha256(input_dir / row["file"]) != row["sha256"]:
                raise RuntimeError("A entrada mudou durante a deduplicação")
        summary = {
            "fingerprint": fingerprint, "parameters": PARAMETERS, "inputs": manifest,
            "input_dir": str(input_dir), "output_dir": str(output_dir), "run_dir": str(run_dir),
            "input_records": len(records), "output_records": len(kept), "removed_records": len(removals),
            **diagnostics, "gpu": gpu,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "nemo_curator_version": importlib.metadata.version("nemo-curator"),
            "implementation_sha256": file_sha256(Path(__file__)),
            "audit_sha256": {
                name: file_sha256(run_dir / name)
                for name in ("records.jsonl", "removals.jsonl")
            },
            "output_sha256": file_sha256(dataset_path),
            "elapsed_seconds": time.perf_counter() - started,
        }
        (run_dir / "manifest.json").write_text(canonical_json(summary) + "\n", encoding="utf-8")
        backup = None
        if output_dir.exists():
            backup_parent = Path(tempfile.mkdtemp(prefix=".fuzzy-backup-", dir=output_dir.parent))
            backup = backup_parent / output_dir.name
            output_dir.rename(backup)
        try:
            staged.rename(output_dir)
        except OSError:
            if backup is not None:
                backup.rename(output_dir)
            raise
        if backup is not None:
            print(f"Output fuzzy anterior preservado em: {backup}")
    print(
        f"Fuzzy: {len(records)} entradas; {len(kept)} representantes; "
        f"{len(removals)} removidos; {summary['elapsed_seconds']:.2f}s.\nAuditoria: {run_dir}"
    )
    return summary
