# Undergraduate Exploratory Practice

## Project Positioning

This project is an undergraduate-stage exploration based on the original SoftSkill
paper and repository. It uses the upstream implementation as a learning and research
base, then adds experiments around shared and task-specific soft-prefix capacity.
It is not the official SoftSkill repository and does not claim authorship of the
original method or paper results.

## Research Question

The exploration asks whether several heterogeneous tasks can share a compact behavior
prefix and still retain task-specific capacity. A stronger version of the question is
whether a jointly trained shared block can help a new task that did not participate in
the joint training.

## Experimental Structure

- Baseline: each task learns an independent 32-token prefix.
- Exploratory layout: one `Shared16` block is shared across tasks and each task owns a
  separate `Task16` block.
- Inference budget: 32 virtual tokens for both layouts.
- Model: the base language model remains frozen.
- Training schedule: 30 optimizer steps update only `Shared16`; the following 120
  optimizer steps jointly update shared and task-specific blocks.
- Behavior initialization: compact behavior descriptions initialize shared evidence
  handling and task-specific answer procedures.

For three tasks, independent prefixes require 96 unique prefix vectors. The shared
layout uses 64 unique vectors, a one-third reduction in stored prefix vectors. This is
a structural count, not proof of a net systems-level efficiency gain.

## Preliminary Observations

The explored `Shared16 + Task16` configuration can remain near the independent
baseline under the tested setup. Behavior-oriented initialization appears relevant,
especially for LiveMath, while residual variants show higher variation across seeds.
These observations motivate further controlled experiments rather than a final claim.

## Evidence Limits

Several factors change at the same time: shared behavior initialization, task behavior
initialization, parameter tying, and the two-stage learning-rate schedule. The current
experiments therefore cannot attribute an observed difference to sharing alone.

The behavior Markdown artifacts also contain validation-influenced selection history.
They should be treated as exploratory evidence, not a leakage-free confirmation. The
current work does not establish that the shared block has learned universal semantics
or that representation similarity implies functional transfer.

## Planned Follow-up

- compare independent, fully shared, and shared/private layouts under matched
  initialization and training schedules;
- restore or interchange shared/task components to test causal contribution;
- test leave-one-task-out transfer and gradient conflict;
- freeze the protocol before multi-seed evaluation on an untouched confirmation set.

The purpose of these follow-ups is to determine when sharing is useful, when it causes
negative transfer, and which conclusions remain valid under controlled comparisons.
