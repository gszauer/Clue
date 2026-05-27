#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from array import array
from collections import Counter
from pathlib import Path
from typing import Iterable


TARGET_VOCAB_SIZE = 512
TRAINING_SUBDIRS = ("location", "weapon", "unsolvable")
ENDING_PREFIXES = (
    "Therefore the murderer is: ",
    "The murderer is ",
    "The murderer was ",
    "The Murderer is ",
    "So the murderer is ",
    "So the murderer was ",
)


def classify_char(ch: str) -> int:
    code = ord(ch)
    if 0x30 <= code <= 0x39:
        return 1
    if 0x41 <= code <= 0x5A:
        return 2
    if 0x61 <= code <= 0x7A:
        return 2
    if code in (0x20, 0x09, 0x0A, 0x0D):
        return 3
    if code > 127:
        return 2
    return 4


def pre_tokenize(text: str) -> Iterable[tuple[int, ...]]:
    """Match model.js Tokenizer._preTokenize exactly."""
    if not text:
        return

    chars: list[str] = []
    current_cat = classify_char(text[0])
    for ch in text:
        cat = classify_char(ch)
        if cat != current_cat:
            yield tuple("".join(chars).encode("utf-8"))
            chars = []
            current_cat = cat
        chars.append(ch)

    yield tuple("".join(chars).encode("utf-8"))


def apply_merge(chunk: tuple[int, ...], a: int, b: int, new_id: int) -> tuple[int, ...]:
    result: list[int] = []
    index = 0
    while index < len(chunk):
        if index < len(chunk) - 1 and chunk[index] == a and chunk[index + 1] == b:
            result.append(new_id)
            index += 2
        else:
            result.append(chunk[index])
            index += 1
    return tuple(result)


class ByteBPE:
    def __init__(self) -> None:
        self.vocab: list[bytes] = [bytes([i]) for i in range(256)]
        # pair -> (rank, new_id)
        self.merges: dict[tuple[int, int], tuple[int, int]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def train_from_chunk_counts(
        self,
        chunk_counts: Counter[tuple[int, ...]],
        target_vocab_size: int,
    ) -> list[dict]:
        history: list[dict] = []

        while len(self.vocab) < target_vocab_size:
            pair_counts: dict[tuple[int, int], int] = {}
            for chunk, freq in chunk_counts.items():
                for index in range(len(chunk) - 1):
                    pair = (chunk[index], chunk[index + 1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + freq

            if not pair_counts:
                break

            best_pair: tuple[int, int] | None = None
            best_count = 0
            for pair, count in pair_counts.items():
                if count > best_count:
                    best_pair = pair
                    best_count = count

            if best_pair is None or best_count < 2:
                break

            a, b = best_pair
            new_id = len(self.vocab)
            rank = len(self.merges)
            self.merges[(a, b)] = (rank, new_id)
            self.vocab.append(self.vocab[a] + self.vocab[b])

            merged_counts: Counter[tuple[int, ...]] = Counter()
            for chunk, freq in chunk_counts.items():
                merged_counts[apply_merge(chunk, a, b, new_id)] += freq
            chunk_counts = merged_counts

            history.append(
                {
                    "rank": rank,
                    "new_id": new_id,
                    "pair": [a, b],
                    "count": best_count,
                    "text": self.vocab[new_id].decode("utf-8", errors="replace"),
                }
            )

        return history

    def encode_chunk(self, chunk: tuple[int, ...]) -> tuple[int, ...]:
        while True:
            best_rank = sys.maxsize
            best_a = -1
            best_b = -1
            best_new_id = -1

            for index in range(len(chunk) - 1):
                merge = self.merges.get((chunk[index], chunk[index + 1]))
                if merge is None:
                    continue
                rank, new_id = merge
                if rank < best_rank:
                    best_rank = rank
                    best_a = chunk[index]
                    best_b = chunk[index + 1]
                    best_new_id = new_id

            if best_rank == sys.maxsize:
                return chunk

            chunk = apply_merge(chunk, best_a, best_b, best_new_id)

    def encode(self, text: str, cache: dict[tuple[int, ...], tuple[int, ...]] | None = None) -> list[int]:
        ids: list[int] = []
        for chunk in pre_tokenize(text):
            if cache is None:
                encoded = self.encode_chunk(chunk)
            else:
                encoded = cache.get(chunk)
                if encoded is None:
                    encoded = self.encode_chunk(chunk)
                    cache[chunk] = encoded
            ids.extend(encoded)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        data = b"".join(self.vocab[int(token_id)] for token_id in ids)
        return data.decode("utf-8", errors="replace")

    def to_json(self) -> str:
        ordered: list[list[int]] = [[0, 0] for _ in range(len(self.merges))]
        for pair, (rank, _new_id) in self.merges.items():
            ordered[rank] = [pair[0], pair[1]]
        # model.js deserializeFromJSON expects exactly this shape.
        return json.dumps({"merges": ordered}, separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> "ByteBPE":
        data = json.loads(text)
        tokenizer = cls()
        tokenizer.vocab = [bytes([i]) for i in range(256)]
        tokenizer.merges = {}

        for pair in data["merges"]:
            a, b = int(pair[0]), int(pair[1])
            new_id = len(tokenizer.vocab)
            tokenizer.merges[(a, b)] = (len(tokenizer.merges), new_id)
            tokenizer.vocab.append(tokenizer.vocab[a] + tokenizer.vocab[b])

        return tokenizer

    @classmethod
    def load(cls, path: Path) -> "ByteBPE":
        return cls.from_json(path.read_text(encoding="utf-8"))

    def save(self, path: Path) -> None:
        path.write_text(self.to_json() + "\n", encoding="utf-8")


def corpus_files(corpus_dir: Path) -> list[Path]:
    files: list[Path] = []
    for subdir in TRAINING_SUBDIRS:
        files.extend(path.resolve() for path in sorted((corpus_dir / subdir).glob("*.txt")))
    return files


def build_chunk_counts(files: list[Path]) -> tuple[Counter[tuple[int, ...]], dict]:
    counts: Counter[tuple[int, ...]] = Counter()
    stats = {"files": len(files), "bytes": 0, "chunks": 0, "unique_chunks": 0}

    for path in files:
        text = path.read_text(encoding="utf-8")
        stats["bytes"] += len(text.encode("utf-8"))
        for chunk in pre_tokenize(text):
            counts[chunk] += 1
            stats["chunks"] += 1

    stats["unique_chunks"] = len(counts)
    return counts, stats


def write_uint16_tokens(path: Path, ids: list[int], append: bool) -> None:
    values = array("H", ids)
    if sys.byteorder != "little":
        values.byteswap()
    mode = "ab" if append else "wb"
    with path.open(mode) as handle:
        values.tofile(handle)


def write_uint8_values(path: Path, values: list[int], append: bool) -> None:
    mode = "ab" if append else "wb"
    with path.open(mode) as handle:
        array("B", values).tofile(handle)


def write_uint32_values(path: Path, values: list[int], append: bool) -> None:
    data = array("I", values)
    if sys.byteorder != "little":
        data.byteswap()
    mode = "ab" if append else "wb"
    with path.open(mode) as handle:
        data.tofile(handle)


def answer_span(block_text: str, tokenizer: ByteBPE, cache: dict[tuple[int, ...], tuple[int, ...]]) -> tuple[int, int] | None:
    lines = block_text.strip().splitlines()
    if not lines:
        return None

    final_line = lines[-1]
    if not final_line.endswith(" FIN."):
        return None

    for prefix in ENDING_PREFIXES:
        if not final_line.startswith(prefix):
            continue
        answer = final_line[len(prefix) : -len(" FIN.")]
        prompt = "\n".join(lines[:-1] + [prefix])
        start = len(tokenizer.encode(prompt, cache))
        width = len(tokenizer.encode(answer, cache))
        return start, width

    return None


def tokenize_files(
    tokenizer: ByteBPE,
    files: list[Path],
    output_path: Path,
    loss_weights_path: Path | None,
    example_index_path: Path | None,
    answer_weight: int,
    root: Path,
) -> tuple[int, int, int, list[dict]]:
    cache: dict[tuple[int, ...], tuple[int, ...]] = {}
    offsets: list[dict] = []
    total_tokens = 0
    answer_weighted_tokens = 0
    example_count = 0

    if output_path.exists():
        output_path.unlink()
    if loss_weights_path is not None and loss_weights_path.exists():
        loss_weights_path.unlink()
    if example_index_path is not None and example_index_path.exists():
        example_index_path.unlink()

    for file_index, path in enumerate(files):
        file_ids: list[int] = []
        file_weights: list[int] = []
        for block in path.read_text(encoding="utf-8").split("\n\n"):
            if not block.strip():
                continue
            block_text = block + "\n\n"
            block_ids = tokenizer.encode(block_text, cache)
            block_weights = [1] * len(block_ids)
            block_start = total_tokens + len(file_ids)

            span = answer_span(block_text, tokenizer, cache)
            if span is not None:
                start, width = span
                for offset in range(start, min(start + width, len(block_weights))):
                    block_weights[offset] = answer_weight
                    answer_weighted_tokens += 1

            file_ids.extend(block_ids)
            file_weights.extend(block_weights)
            if example_index_path is not None:
                write_uint32_values(
                    example_index_path,
                    [block_start, len(block_ids)],
                    append=example_count > 0,
                )
            example_count += 1

        ids = file_ids
        write_uint16_tokens(output_path, ids, append=file_index > 0)
        if loss_weights_path is not None:
            write_uint8_values(loss_weights_path, file_weights, append=file_index > 0)
        offsets.append(
            {
                "path": str(path.relative_to(root)),
                "start_token": total_tokens,
                "token_count": len(ids),
            }
        )
        total_tokens += len(ids)

    return total_tokens, answer_weighted_tokens, example_count, offsets


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the JS-compatible byte-BPE tokenizer.")
    parser.add_argument("--corpus-dir", type=Path, default=Path("corpus"))
    parser.add_argument("--vocab-size", type=int, default=TARGET_VOCAB_SIZE)
    parser.add_argument("--vocab-out", type=Path, default=Path("vocab.json"))
    parser.add_argument("--tokens-out", type=Path, default=Path("generate_working/tokenized_corpus_u16le.bin"))
    parser.add_argument("--loss-weights-out", type=Path, default=Path("generate_working/loss_weights_u8.bin"))
    parser.add_argument("--example-index-out", type=Path, default=Path("generate_working/example_index_u32le.bin"))
    parser.add_argument("--answer-weight", type=int, default=32)
    parser.add_argument("--report-out", type=Path, default=Path("generate_working/tokenization_report.json"))
    args = parser.parse_args()

    root = Path.cwd().resolve()
    files = corpus_files(args.corpus_dir)
    if not files:
        raise SystemExit(f"No corpus files found under {args.corpus_dir}")

    chunk_counts, corpus_stats = build_chunk_counts(files)
    tokenizer = ByteBPE()
    merge_history = tokenizer.train_from_chunk_counts(chunk_counts, args.vocab_size)
    tokenizer.save(args.vocab_out)

    if not 1 <= args.answer_weight <= 255:
        raise SystemExit("--answer-weight must fit in uint8: 1..255")

    token_count, answer_weighted_tokens, example_count, offsets = tokenize_files(
        tokenizer,
        files,
        args.tokens_out,
        args.loss_weights_out,
        args.example_index_out,
        args.answer_weight,
        root,
    )
    report = {
        "tokenizer": {
            "type": "byte_bpe",
            "target_vocab_size": args.vocab_size,
            "actual_vocab_size": tokenizer.vocab_size,
            "base_byte_tokens": 256,
            "learned_merges": len(tokenizer.merges),
            "compatible_with_model_js_serializer": True,
        },
        "corpus": corpus_stats,
        "tokenized_output": {
            "path": str(args.tokens_out),
            "dtype": "uint16 little-endian",
            "token_count": token_count,
            "bytes": args.tokens_out.stat().st_size,
            "file_offsets": offsets,
        },
        "loss_weights": {
            "path": str(args.loss_weights_out),
            "dtype": "uint8",
            "default_weight": 1,
            "answer_weight": args.answer_weight,
            "answer_weighted_tokens": answer_weighted_tokens,
            "bytes": args.loss_weights_out.stat().st_size,
        },
        "example_index": {
            "path": str(args.example_index_out),
            "dtype": "uint32 little-endian pairs of start_token, token_count",
            "examples": example_count,
            "bytes": args.example_index_out.stat().st_size,
        },
        "merge_history": merge_history,
    }
    args.report_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {args.vocab_out}")
    print(f"wrote {args.tokens_out}")
    print(f"vocab_size: {tokenizer.vocab_size}")
    print(f"tokens: {token_count}")
    print(f"tokenized_bytes: {args.tokens_out.stat().st_size}")
    print(f"wrote {args.loss_weights_out}")
    print(f"answer_weighted_tokens: {answer_weighted_tokens}")
    print(f"wrote {args.example_index_out}")
    print(f"examples: {example_count}")


if __name__ == "__main__":
    main()
