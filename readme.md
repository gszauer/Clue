# Clue 250K

Clue 250K is a tiny educational language model trained to solve synthetic
Clue-style murder mysteries. It is intentionally small enough to understand
end-to-end: the tokenizer, model, training loop, inference code, JavaScript
reference implementation, weights, and browser demo are all in this repository.

The model is not a general-purpose LLM. It is a constrained toy model trained
on a fixed vocabulary of names, rooms, weapons, and wounds. Within that toy
domain, it learns to complete prompts like:

```text
Josh is in the library
Basil is in the kitchen
A body was found in the library
Maria is in the living room
Therefore the murderer is:
```

with:

```text
Josh FIN.
```

`FIN.` is ordinary text, not a special token.

## Repository Layout

- `index.html` - single-file mobile-friendly canvas demo.
- `model.js` - readable JavaScript reference model and tokenizer.
- `tokenizer.py` - trains and serializes the byte-BPE tokenizer.
- `trainer.py` - MLX training script for Apple Silicon.
- `inference.py` - Python inference and validation script.
- `vocab.json` - 512-token byte-BPE vocabulary/merge table.
- `weights.bin` - trained model weights, serialized for `model.js`.

The synthetic corpus and working artifacts are not committed to this repo. The corpus is hosted at:

https://huggingface.co/datasets/gszauer/Clue250K

## Model

Clue 250K is a decoder-only transformer with a deliberately small shape:

- Vocabulary size: `512`
- Context length: `96` tokens
- Transformer blocks: `3`
- Feature dimension: `72`
- MLP hidden dimension: `288` (`4x`)
- Attention: one causal full-width attention head per block
- Normalization: RMSNorm
- Activation: ReLU
- Positional embeddings: learned
- Output projection: tied token embedding matrix
- Parameters: `230,904`
- Weight file: `weights.bin`, raw little-endian `float32`

The exact parameter count is lower than the rough "250K" name because the
model ties token embeddings and unembeddings.

The tokenizer is a byte-level BPE tokenizer:

- 256 base byte tokens
- 256 learned merges
- 512 total tokens
- Serialized as `vocab.json`
- Compatible with `model.js` `Tokenizer.deserializeFromJSON`

## Training Data

The final corpus is synthetic and deterministic from seed `20260526`.

- Size: `500.0 MiB`
- Examples: `3,164,051`
- Byte-BPE tokens: `240,244,946`
- Files generated: `318`
- Names: `64`
- Locations: `24`
- Weapons: `20`

Most examples use the canonical ending:

```text
Therefore the murderer is: {answer} FIN.
```

Some examples use alternate endings so the model does not only memorize one
final-line shape:

```text
The murderer is {answer} FIN.
The murderer was {answer} FIN.
The Murderer is {answer} FIN.
So the murderer is {answer} FIN.
So the murderer was {answer} FIN.
```

The canonical `Therefore the murderer is:` ending is still dominant, at about
70% of the generated corpus.

## Training Run

The included `weights.bin` was trained with `trainer.py` on MLX.

- Corpus tokens per epoch: `240,244,946`
- Epochs: `4`
- Total token presentations: `960,979,784`
- Tokens per parameter per epoch: `1,040.45`
- Total token presentations per parameter: `4,161.82`
- Batch size: `1024`
- Steps: `12,360`
- Learning rate: `0.001`
- Weight decay: `0`
- Gradient clipping: `1.0`
- Seed: `20260526`
- Initial loss: `6.1790`
- Final loss: `0.2469`

The tokenization step also writes per-token loss weights. Normal tokens have
weight `1`; answer tokens have weight `32`, which biases training toward
getting the murderer name right. Training uses example-aligned batches from
`example_index_u32le.bin` rather than arbitrary stream chunks.

A sampled greedy validation run over 100 held-out-style corpus examples got
`100/100` exact murderer-name completions.

## Supported Prompt Formats

The model is best at prompts that match the synthetic corpus.

### Location Mystery

Use one line per person:

```text
Josh is in the library
Basil is in the kitchen
Maria is in the living room
A body was found in the library
Therefore the murderer is:
```

The answer is the unique person in the room where the body was found.

If no person is in that room, or multiple people are in that room, the answer is:

```text
Unknown FIN.
```

### Weapon Mystery

The compact weapon format is preferred because the model only has a 96-token
context:

```text
Josh has a pipe
Basil has a knife
Maria has a rope
A body was found in the library with a knife wound
Therefore the murderer is:
```

The answer is the unique person whose weapon matches the wound.

The corpus also contains mixed lines like:

```text
Josh is in the library with a pipe
```

but four long person lines can run out the 96-token context. The browser demo
therefore emits the shorter `Name has a weapon` form for weapon mysteries.

If no weapon matches the wound, or multiple people have matching weapons, the
answer is:

```text
Unknown FIN.
```

### Generation From a Seed

The model can also continue a seed, for example:

```bash
python3 inference.py "Maria is"
```

This asks the model to generate a whole short mystery until `FIN.` or until the
96-token context fills.

## Running

Start the browser demo from the repo root:

```bash
python3 -m http.server 8000 --bind 0.0.0.0
```

Then open:

```text
http://localhost:8000/index.html
```

Run Python inference:

```bash
python3 inference.py "Maria is"
```

Run a deterministic solve:

```bash
python3 inference.py $'Josh is in the library\nBasil is in the kitchen\nA body was found in the library\nMaria is in the living room\nTherefore the murderer is: ' --temperature 0 --top-k 0
```

Validate against a local corpus checkout:

```bash
python3 inference.py --validate 100 --temperature 0 --top-k 0 --corpus-dir corpus
```

## Training From the Dataset

After downloading the Hugging Face dataset into `corpus/`, regenerate the
tokenizer and tokenized training files:

```bash
python3 tokenizer.py --corpus-dir corpus
```

Train for four epochs:

```bash
python3 trainer.py --epochs 4
```

This writes:

- `vocab.json`
- `weights.bin`
- `generate_working/tokenized_corpus_u16le.bin`
- `generate_working/loss_weights_u8.bin`
- `generate_working/example_index_u32le.bin`
- tokenization and training reports under `generate_working/`

`generate_working/` is scratch space and does not need to be committed.

## Valid Names

```text
Alex, Avery, Basil, Blair, Blake, Casey, Dana, Devin, Drew, Eden, Ellis,
Emery, Erin, Felix, Finn, Gail, Gray, Harper, Hazel, Ira, Jamie, Jordan,
Josh, Jules, Kai, Kendall, Lane, Laurel, Lee, Logan, Mara, Maria, Micah,
Morgan, Nico, Noel, Nora, Owen, Paige, Parker, Peyton, Quinn, Reese, Remy,
Riley, Robin, Rowan, Sage, Sam, Sasha, Sidney, Sky, Sloan, Talia, Taylor,
Theo, Vale, Wren, Zane, Mina, Ivy, Lena, Toby, June
```

## Valid Locations

```text
library, kitchen, living room, study, hallway, cellar, attic, garden, garage,
pantry, office, lounge, chapel, conservatory, ballroom, dining room, parlor,
nursery, balcony, courtyard, observatory, gallery, theater, basement
```

## Valid Weapons

```text
knife, pipe, rope, wrench, candle, poison, revolver, hammer, axe, poker,
dagger, chain, bat, statue, scissors, shovel, vial, club, cable, needle
```

## Valid Wounds

Weapon mysteries map wounds back to weapons:

```text
knife     -> knife wound
pipe      -> pipe bruise
rope      -> rope mark
wrench    -> wrench bruise
candle    -> candle wax
poison    -> poison trace
revolver  -> bullet wound
hammer    -> hammer dent
axe       -> axe wound
poker     -> poker burn
dagger    -> dagger wound
chain     -> chain mark
bat       -> bat bruise
statue    -> statue blow
scissors  -> scissor cut
shovel    -> shovel mark
vial      -> glass cut
club      -> club bruise
cable     -> cable mark
needle    -> needle mark
```

## Notes

- This model is intentionally overtrained for the tiny synthetic domain.
- It relies on the controlled vocabulary above.
- It should be evaluated as a learning/demo model, not as a robust general
  reasoning system.
- `weights.bin` and `vocab.json` are the only trained artifacts needed by the
  browser demo.
