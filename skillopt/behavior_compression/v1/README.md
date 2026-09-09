# Behavior-compression initialization v1

These Markdown files factor the three optimized task skills into one shared
behavior and one private behavior per task.  The factorization is semantic,
not a literal sentence or token intersection.

| Prefix block | Behavior represented |
| --- | --- |
| `shared_behavior.md` | Evidence grounding, answer-constraint checking, and exact supported output |
| `searchqa_behavior.md` | Retrieval-noise filtering, entity/relation-direction checking, and concise unambiguous answers |
| `livemath_behavior.md` | Option comparison across assumptions, quantifiers, endpoints, and statement strength |
| `docvqa_behavior.md` | Label/spatial-layout field lookup and exact minimal-span extraction |

Each file starts directly with its behavior-bearing seed sentence.  There is
no title or placeholder before it because direct SoftSkill initialization
uses the first `prefix_length` tokenizer embeddings rather than a contextual
encoding of the complete document.

The source artifacts used for the manual factorization are:

- `ckpt/searchqa/gpt5.5_skill.md`
- `ckpt/livemath/gpt5.5_skill.md`
- `ckpt/docvqa/gpt5.5_skill.md`

All four stripped texts are exactly 16 tokens under the Qwen/Qwen3.5-4B
tokenizer. `tokenizer_audit.json` records the raw text, source and selected
token IDs, counts, decoded selection, and repetition/truncation flags. The
training entrypoint independently repeats that audit in `protocol_audit.json`.

`provenance.json` records the source audit. The released checkpoints lack
generation logs, while the current SkillOpt generator performs candidate
selection on `valid_seen`; consequently this v1 run must be reported as
validation-influenced rather than a certified train-only result.
