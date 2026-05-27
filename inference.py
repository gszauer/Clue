#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import mlx.core as mx
import numpy as np

from tokenizer import ByteBPE
from trainer import CONFIG, MinimalGPT, load_weights_bin


ENDING_PREFIXES = [
    "Therefore the murderer is: ",
    "The murderer is ",
    "The murderer was ",
    "The Murderer is ",
    "So the murderer is ",
    "So the murderer was ",
]


def load_model(weights_path: Path) -> MinimalGPT:
    model = MinimalGPT(CONFIG)
    loaded = load_weights_bin(model, weights_path)
    expected = sum(int(np.prod(array.shape)) for array in model.js_ordered_arrays())
    if loaded != expected:
        raise ValueError(f"loaded {loaded} params, expected {expected}")
    mx.eval(model.parameters())
    model.eval()
    return model


def next_token(model: MinimalGPT, ids: list[int], temperature: float) -> int:
    return sample_next_token(model, ids, temperature, top_k=0, top_p=0.0)


def filtered_logits(logits: mx.array, top_k: int, top_p: float) -> mx.array:
    values = np.asarray(logits, dtype=np.float32)

    if top_k > 0 and top_k < values.size:
        keep = np.argpartition(values, -top_k)[-top_k:]
        mask = np.full(values.shape, -np.inf, dtype=np.float32)
        mask[keep] = values[keep]
        values = mask

    if 0.0 < top_p < 1.0:
        finite = np.isfinite(values)
        if finite.any():
            candidate_indices = np.where(finite)[0]
            candidate_logits = values[candidate_indices]
            order = np.argsort(candidate_logits)[::-1]
            sorted_indices = candidate_indices[order]
            sorted_logits = candidate_logits[order]
            shifted = sorted_logits - sorted_logits.max()
            probs = np.exp(shifted)
            probs /= probs.sum()
            cumulative = np.cumsum(probs)
            keep_count = int(np.searchsorted(cumulative, top_p, side="left")) + 1
            keep_indices = sorted_indices[:keep_count]
            mask = np.full(values.shape, -np.inf, dtype=np.float32)
            mask[keep_indices] = values[keep_indices]
            values = mask

    return mx.array(values, dtype=mx.float32)


def sample_next_token(
    model: MinimalGPT,
    ids: list[int],
    temperature: float,
    top_k: int,
    top_p: float,
) -> int:
    context_ids = ids[-CONFIG.max_context :]
    x = mx.array([context_ids], dtype=mx.int32)
    logits = model(x)[0, -1]

    if temperature <= 0:
        token_id = mx.argmax(logits)
    else:
        logits = filtered_logits(logits / temperature, top_k, top_p)
        token_id = mx.random.categorical(logits)

    mx.eval(token_id)
    return int(token_id.item())


def generate(
    model: MinimalGPT,
    tokenizer: ByteBPE,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    stop_on_fin: bool,
) -> str:
    ids = tokenizer.encode(prompt)
    if not ids:
        raise ValueError("prompt must encode to at least one token")

    for _ in range(max_new_tokens):
        if len(ids) >= CONFIG.max_context:
            break
        ids.append(sample_next_token(model, ids, temperature, top_k, top_p))
        if stop_on_fin and "FIN." in tokenizer.decode(ids):
            break

    return tokenizer.decode(ids)


def corpus_files(corpus_dir: Path) -> list[Path]:
    files: list[Path] = []
    for subdir in ("location", "weapon", "unsolvable"):
        files.extend(sorted((corpus_dir / subdir).glob("*.txt")))
    return files


def prompt_answer_from_block(block: str) -> tuple[str, str] | None:
    lines = block.strip().splitlines()
    if not lines:
        return None

    final_line = lines[-1]
    if not final_line.endswith(" FIN."):
        return None

    for prefix in ENDING_PREFIXES:
        if final_line.startswith(prefix):
            answer = final_line[len(prefix) : -len(" FIN.")]
            prompt = "\n".join(lines[:-1] + [prefix])
            return prompt, answer

    return None


def sample_validation_blocks(corpus_dir: Path, count: int, seed: int) -> list[str]:
    files = corpus_files(corpus_dir)
    if not files:
        raise ValueError(f"no corpus files found under {corpus_dir}")

    rng = random.Random(seed)
    blocks: list[str] = []
    while len(blocks) < count:
        path = rng.choice(files)
        candidates = [block for block in path.read_text(encoding="utf-8").split("\n\n") if block.strip()]
        blocks.append(rng.choice(candidates))
    return blocks


def predicted_answer(prompt: str, generated: str) -> str:
    completion = generated[len(prompt) :]
    match = re.match(r"\s*([A-Za-z]+)", completion)
    return match.group(1) if match else ""


def validate(
    model: MinimalGPT,
    tokenizer: ByteBPE,
    corpus_dir: Path,
    count: int,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> None:
    blocks = sample_validation_blocks(corpus_dir, count, seed)
    correct = 0
    attempted = 0
    failures: list[tuple[str, str, str]] = []

    for block in blocks:
        parsed = prompt_answer_from_block(block)
        if parsed is None:
            continue
        prompt, expected = parsed
        generated = generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_on_fin=True,
        )
        pred = predicted_answer(prompt, generated)
        attempted += 1
        if pred == expected:
            correct += 1
        elif len(failures) < 5:
            failures.append((expected, pred, generated))

    accuracy = correct / attempted if attempted else 0.0
    print(f"validation_attempted: {attempted}")
    print(f"validation_correct: {correct}")
    print(f"validation_accuracy: {accuracy:.3f}")

    for index, (expected, pred, generated) in enumerate(failures, start=1):
        print(f"\nmiss {index}: expected={expected} predicted={pred}")
        print(generated)


def interactive_loop(
    model: MinimalGPT,
    tokenizer: ByteBPE,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    stop_on_fin: bool,
) -> None:
    print("Enter a prompt. Empty input exits.")
    while True:
        prompt = input("> ")
        if not prompt:
            break
        print(generate(model, tokenizer, prompt, max_new_tokens, temperature, top_k, top_p, stop_on_fin))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference or validation for the Clue100 model.")
    parser.add_argument("prompt_text", nargs="*", help="Prompt text. Example: python3 inference.py 'Maria is'")
    parser.add_argument("--weights", type=Path, default=Path("weights.bin"))
    parser.add_argument("--vocab", type=Path, default=Path("vocab.json"))
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=0,
        help="0 means generate until FIN. or until the 96-token context is full.",
    )
    parser.add_argument("--temperature", type=float, default=0.8, help="0 means greedy argmax.")
    parser.add_argument("--top-k", type=int, default=20, help="0 disables top-k filtering.")
    parser.add_argument("--top-p", type=float, default=0.0, help="0 disables top-p filtering.")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--no-stop-on-fin", action="store_true")
    parser.add_argument("--validate", type=int, default=0, help="Sample this many corpus examples.")
    parser.add_argument("--corpus-dir", type=Path, default=Path("corpus"))
    parser.add_argument("--seed", type=int, default=20260526)
    args = parser.parse_args()

    if args.seed:
        mx.random.seed(args.seed)
    if args.temperature < 0:
        raise SystemExit("--temperature must be >= 0")
    if args.top_k < 0:
        raise SystemExit("--top-k must be >= 0")
    if not 0.0 <= args.top_p <= 1.0:
        raise SystemExit("--top-p must be in [0, 1]")
    if not args.vocab.exists():
        raise SystemExit(f"{args.vocab} is missing. Run `python3 tokenizer.py` first.")
    if not args.weights.exists():
        raise SystemExit(f"{args.weights} is missing. Run `python3 trainer.py` first.")

    tokenizer = ByteBPE.load(args.vocab)
    model = load_model(args.weights)
    prompt = args.prompt if args.prompt else " ".join(args.prompt_text)
    max_new_tokens = args.max_new_tokens
    if max_new_tokens == 0:
        max_new_tokens = CONFIG.max_context

    if args.validate > 0:
        validate(
            model,
            tokenizer,
            args.corpus_dir,
            args.validate,
            args.seed,
            max_new_tokens,
            args.temperature,
            args.top_k,
            args.top_p,
        )
    elif prompt:
        for index in range(args.samples):
            if args.samples > 1:
                print(f"--- sample {index + 1} ---")
            print(
                generate(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens,
                    args.temperature,
                    args.top_k,
                    args.top_p,
                    stop_on_fin=not args.no_stop_on_fin,
                )
            )
    else:
        interactive_loop(
            model,
            tokenizer,
            max_new_tokens,
            args.temperature,
            args.top_k,
            args.top_p,
            stop_on_fin=not args.no_stop_on_fin,
        )


if __name__ == "__main__":
    main()
