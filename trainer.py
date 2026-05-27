#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 512
    max_context: int = 96
    num_blocks: int = 3
    feature_dim: int = 72
    mlp_multiplier: int = 4
    rms_epsilon: float = 1e-5
    tied_unembedding: bool = True


CONFIG = ModelConfig()


def small_random(rows: int, columns: int) -> mx.array:
    scale = 1.0 / math.sqrt(rows)
    return mx.random.uniform(-scale, scale, shape=(rows, columns), dtype=mx.float32)


class RMSNorm(nn.Module):
    def __init__(self, feature_dim: int, epsilon: float) -> None:
        super().__init__()
        self.learnedGamma = mx.ones((feature_dim,), dtype=mx.float32)
        self.epsilon = epsilon

    def __call__(self, x: mx.array) -> mx.array:
        rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.epsilon)
        return (x / rms) * self.learnedGamma


class AttentionHeadSingle(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.learnedQ = small_random(feature_dim, feature_dim)
        self.learnedK = small_random(feature_dim, feature_dim)
        self.learnedV = small_random(feature_dim, feature_dim)
        self.learnedO = small_random(feature_dim, feature_dim)
        self.head_dim = feature_dim

    def __call__(self, x: mx.array) -> mx.array:
        q = x @ self.learnedQ
        k = x @ self.learnedK
        v = x @ self.learnedV

        scores = (q @ mx.swapaxes(k, -1, -2)) * (1.0 / math.sqrt(self.head_dim))
        sequence_length = x.shape[1]
        mask = mx.triu(mx.ones((sequence_length, sequence_length), dtype=mx.bool_), k=1)
        scores = mx.where(mask[None, :, :], -1e9, scores)
        probabilities = mx.softmax(scores, axis=-1)
        return (probabilities @ v) @ self.learnedO


class Perceptron(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.learnedUp = small_random(feature_dim, hidden_dim)
        self.learnedDown = small_random(hidden_dim, feature_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return mx.maximum(x @ self.learnedUp, 0.0) @ self.learnedDown


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.layerNorm1 = RMSNorm(config.feature_dim, config.rms_epsilon)
        self.attention = AttentionHeadSingle(config.feature_dim)
        self.layerNorm2 = RMSNorm(config.feature_dim, config.rms_epsilon)
        self.mlp = Perceptron(config.feature_dim, config.feature_dim * config.mlp_multiplier)

    def __call__(self, x: mx.array) -> mx.array:
        after_attention = x + self.attention(self.layerNorm1(x))
        return after_attention + self.mlp(self.layerNorm2(after_attention))


class MinimalGPT(nn.Module):
    def __init__(self, config: ModelConfig = CONFIG) -> None:
        super().__init__()
        self.config = config
        self.tokenEmbeddings = small_random(config.vocab_size, config.feature_dim)
        self.positionalEmbeddings = small_random(config.max_context, config.feature_dim)
        self.blocks = [TransformerBlock(config) for _ in range(config.num_blocks)]
        self.finalNorm = RMSNorm(config.feature_dim, config.rms_epsilon)

    def __call__(self, token_ids: mx.array) -> mx.array:
        sequence_length = token_ids.shape[1]
        if sequence_length > self.config.max_context:
            raise ValueError(f"input length {sequence_length} exceeds max_context {self.config.max_context}")

        token_emb = self.tokenEmbeddings[token_ids]
        position_emb = self.positionalEmbeddings[:sequence_length]
        x = token_emb + position_emb[None, :, :]
        for block in self.blocks:
            x = block(x)
        x = self.finalNorm(x)
        return x @ mx.swapaxes(self.tokenEmbeddings, 0, 1)

    def js_ordered_arrays(self) -> list[mx.array]:
        arrays: list[mx.array] = [self.tokenEmbeddings, self.positionalEmbeddings]
        for block in self.blocks:
            arrays.extend(
                [
                    block.layerNorm1.learnedGamma,
                    block.attention.learnedQ,
                    block.attention.learnedK,
                    block.attention.learnedV,
                    block.attention.learnedO,
                    block.layerNorm2.learnedGamma,
                    block.mlp.learnedUp,
                    block.mlp.learnedDown,
                ]
            )
        arrays.append(self.finalNorm.learnedGamma)
        return arrays


def exact_param_count(config: ModelConfig = CONFIG) -> int:
    hidden_dim = config.feature_dim * config.mlp_multiplier
    token_embeddings = config.vocab_size * config.feature_dim
    positional_embeddings = config.max_context * config.feature_dim
    attention = 4 * config.feature_dim * config.feature_dim
    mlp = config.feature_dim * hidden_dim + hidden_dim * config.feature_dim
    norms_per_block = 2 * config.feature_dim
    final_norm = config.feature_dim
    untied_unembedding = 0 if config.tied_unembedding else config.feature_dim * config.vocab_size
    return (
        token_embeddings
        + positional_embeddings
        + config.num_blocks * (attention + mlp + norms_per_block)
        + final_norm
        + untied_unembedding
    )


def save_weights_bin(model: MinimalGPT, path: Path) -> int:
    arrays = model.js_ordered_arrays()
    total = 0
    with path.open("wb") as handle:
        for value in arrays:
            data = np.asarray(value, dtype=np.float32).reshape(-1)
            if data.dtype.byteorder == ">":
                data = data.byteswap().newbyteorder("<")
            handle.write(data.astype("<f4", copy=False).tobytes(order="C"))
            total += data.size
    return total


def load_weights_bin(model: MinimalGPT, path: Path) -> int:
    weights = np.fromfile(path, dtype="<f4")
    offset = 0

    for target in model.js_ordered_arrays():
        size = math.prod(target.shape)
        if offset + size > weights.size:
            raise ValueError(f"{path} ended early while loading weights")
        values = weights[offset : offset + size].reshape(tuple(target.shape))
        target[:] = mx.array(values, dtype=mx.float32)
        offset += size

    if offset != weights.size:
        raise ValueError(f"{path} has {weights.size - offset} extra float32 values")
    return offset


def batch_from_stream_memmap(
    tokens: np.memmap,
    loss_weights: np.memmap | None,
    starts: np.ndarray,
    context: int,
) -> tuple[mx.array, mx.array, mx.array]:
    offsets = starts[:, None] + np.arange(context + 1, dtype=np.int64)[None, :]
    batch = np.asarray(tokens[offsets], dtype=np.int32)
    if loss_weights is None:
        weights = np.ones((len(starts), context), dtype=np.float32)
    else:
        weights = np.asarray(loss_weights[offsets[:, 1:]], dtype=np.float32)
    return mx.array(batch[:, :-1]), mx.array(batch[:, 1:]), mx.array(weights)


def batch_from_examples(
    tokens: np.memmap,
    loss_weights: np.memmap | None,
    example_pairs: np.ndarray,
    context: int,
) -> tuple[mx.array, mx.array, mx.array]:
    batch_size = len(example_pairs)
    inputs = np.zeros((batch_size, context), dtype=np.int32)
    targets = np.zeros((batch_size, context), dtype=np.int32)
    weights = np.zeros((batch_size, context), dtype=np.float32)

    for row, (start, length) in enumerate(example_pairs):
        start = int(start)
        length = min(int(length), context + 1)
        if length < 2:
            continue
        end = start + length
        input_width = length - 1
        ids = np.asarray(tokens[start:end], dtype=np.int32)
        inputs[row, :input_width] = ids[:-1]
        targets[row, :input_width] = ids[1:]
        if loss_weights is None:
            weights[row, :input_width] = 1.0
        else:
            weights[row, :input_width] = np.asarray(loss_weights[start + 1 : end], dtype=np.float32)

    return mx.array(inputs), mx.array(targets), mx.array(weights)


def make_loss_fn(vocab_size: int):
    def loss_fn(model: MinimalGPT, inputs: mx.array, targets: mx.array, weights: mx.array) -> mx.array:
        logits = model(inputs)
        losses = nn.losses.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1),
            reduction="none",
        ).reshape(targets.shape)
        weights = weights.astype(mx.float32)
        return mx.sum(losses * weights) / mx.sum(weights)

    return loss_fn


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the tiny Clue GPT with MLX.")
    parser.add_argument("--tokens", type=Path, default=Path("generate_working/tokenized_corpus_u16le.bin"))
    parser.add_argument("--loss-weights", type=Path, default=Path("generate_working/loss_weights_u8.bin"))
    parser.add_argument("--example-index", type=Path, default=Path("generate_working/example_index_u32le.bin"))
    parser.add_argument("--stream-mode", action="store_true", help="Train on fixed stream chunks instead of examples.")
    parser.add_argument("--ignore-loss-weights", action="store_true")
    parser.add_argument("--weights-out", type=Path, default=Path("weights.bin"))
    parser.add_argument("--report-out", type=Path, default=Path("generate_working/training_report.json"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means train the full epoch.")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if not args.tokens.exists():
        raise SystemExit(f"{args.tokens} is missing. Run `python3 tokenizer.py` first.")

    mx.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    tokens = np.memmap(args.tokens, dtype="<u2", mode="r")
    if int(tokens.max()) >= CONFIG.vocab_size:
        raise SystemExit(f"token id exceeds vocab size {CONFIG.vocab_size}")

    loss_weights = None
    if not args.ignore_loss_weights and args.loss_weights.exists():
        loss_weights = np.memmap(args.loss_weights, dtype=np.uint8, mode="r")
        if loss_weights.size != tokens.size:
            raise SystemExit(
                f"{args.loss_weights} has {loss_weights.size} weights, expected {tokens.size}"
            )

    context = CONFIG.max_context
    if tokens.size < context + 1:
        raise SystemExit("token stream is too short")

    example_index = None
    train_items: np.ndarray
    if not args.stream_mode and args.example_index.exists():
        raw_index = np.memmap(args.example_index, dtype="<u4", mode="r")
        if raw_index.size % 2 != 0:
            raise SystemExit(f"{args.example_index} must contain start,length pairs")
        example_index = raw_index.reshape(-1, 2)
        train_items = np.arange(example_index.shape[0], dtype=np.int64)
        mode = "examples"
    else:
        train_items = np.arange(0, tokens.size - context, context, dtype=np.int64)
        mode = "stream"

    steps_per_epoch = math.ceil(len(train_items) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)

    model = MinimalGPT(CONFIG)
    mx.eval(model.parameters())
    optimizer = optim.AdamW(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        bias_correction=True,
    )

    loss_and_grad = nn.value_and_grad(model, make_loss_fn(CONFIG.vocab_size))

    def train_step(inputs: mx.array, targets: mx.array, weights: mx.array) -> mx.array:
        loss, grads = loss_and_grad(model, inputs, targets, weights)
        if args.grad_clip > 0:
            grads, _norm = optim.clip_grad_norm(grads, args.grad_clip)
        optimizer.update(model, grads)
        return loss

    step_fn = train_step
    if args.compile:
        state = [model.state, optimizer.state]
        step_fn = mx.compile(train_step, inputs=state, outputs=state)

    expected_params = exact_param_count(CONFIG)
    actual_params = sum(math.prod(array.shape) for array in model.js_ordered_arrays())
    if actual_params != expected_params:
        raise SystemExit(f"parameter count mismatch: {actual_params} != {expected_params}")

    print("training Clue100 model")
    print(f"tokens: {tokens.size}")
    print(f"mode: {mode}")
    print(f"sequences: {len(train_items)}")
    print(f"batch_size: {args.batch_size}")
    print(f"steps_per_epoch: {steps_per_epoch}")
    print(f"epochs: {args.epochs}")
    print(f"max_steps: {args.max_steps if args.max_steps else 'full'}")
    print(f"params: {actual_params}")
    print(f"loss_weights: {args.loss_weights if loss_weights is not None else 'none'}")

    losses: list[float] = []
    start_time = time.time()
    global_step = 0

    for epoch in range(args.epochs):
        order = rng.permutation(train_items)
        for batch_start in range(0, len(order), args.batch_size):
            if args.max_steps > 0 and global_step >= args.max_steps:
                break

            batch_items = order[batch_start : batch_start + args.batch_size]
            if example_index is None:
                inputs, targets, weights = batch_from_stream_memmap(tokens, loss_weights, batch_items, context)
            else:
                inputs, targets, weights = batch_from_examples(tokens, loss_weights, example_index[batch_items], context)
            loss = step_fn(inputs, targets, weights)
            mx.eval(loss, model.state, optimizer.state)
            loss_value = float(loss.item())
            losses.append(loss_value)
            global_step += 1

            if global_step == 1 or global_step % args.log_interval == 0:
                elapsed = time.time() - start_time
                tokens_seen = global_step * args.batch_size * context
                print(
                    f"step {global_step}/{total_steps} "
                    f"epoch {epoch + 1}/{args.epochs} "
                    f"loss {loss_value:.4f} "
                    f"tokens/s {tokens_seen / max(elapsed, 1e-9):.0f}",
                    flush=True,
                )

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    saved_params = save_weights_bin(model, args.weights_out)
    elapsed = time.time() - start_time
    report = {
        "config": asdict(CONFIG),
        "tokens": int(tokens.size),
        "mode": mode,
        "sequences": int(len(train_items)),
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "steps_completed": global_step,
        "params": saved_params,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "seed": args.seed,
        "elapsed_seconds": elapsed,
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "mean_loss": float(np.mean(losses)) if losses else None,
        "weights_out": str(args.weights_out),
    }
    args.report_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"saved {args.weights_out} ({saved_params} float32 params)")
    print(f"final_loss: {report['final_loss']:.4f}")
    print(f"elapsed_seconds: {elapsed:.1f}")


if __name__ == "__main__":
    main()
