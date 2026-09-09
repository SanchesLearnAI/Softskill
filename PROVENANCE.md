# Code Provenance

This document separates upstream work, local research changes, and third-party code.
It is intended to prevent the derivative repository from being mistaken for the
official SoftSkill implementation or for wholly original work by its maintainer.

## Upstream SoftSkill baseline

- Repository: <https://github.com/xijia-tao/SoftSkill>
- Baseline commit: `4fc53008da110f354746bf36966dc0a2f44d3b92`
- Commit author recorded by Git: Xijia Tao
- Associated paper: *SoftSkill: Behavioral Compression for Contextual Adaptation*
- Paper authors: Xijia Tao, Yihua Teng, Xinyu Fu, Ziru Liu, Kecheng Chen, Yuzhi Zhao, Suiyun Zhang, Rui Liu, and Lingpeng Kong
- Paper identifier: arXiv:2606.20333

Files classified as `upstream_unchanged` in `CODE_PROVENANCE.csv` match paths tracked
by this baseline and have no working-tree modification relative to it. Files classified
as `upstream_modified_locally` existed in the baseline but contain local changes.

## Undergraduate exploratory extension

The local work is presented as exploratory practice conducted during undergraduate
study. This description indicates the educational stage and limited evidentiary status;
it does not override the original authorship and licensing statements above.

Files classified as `local_addition` were absent from the baseline commit. The main
extension areas are:

- shared/private fixed-budget soft-prefix layouts and multitask training;
- prefix-state export, similarity analysis, and causal interventions;
- behavior-compression initialization and associated provenance/tokenizer audits;
- ChartQA and DROP adapters and new-task transfer protocols;
- Slurm launchers, smoke checks, and tests for those extensions.

`local_addition` means only “not present in the selected upstream commit.” It does not
prove that one individual personally authored the file. Before a public release, the
maintainer should add every actual contributor and confirm laboratory, employer,
sponsor, and collaborator publication rights.

## Third-party code

Some files predate the local extension and are themselves derived from other projects.
These are classified as `third_party_vendored_or_adapted` where the boundary can be
identified from in-file notices. See `THIRD_PARTY_NOTICES.md`.

## How the classification was produced

The per-file table compares the working tree with `upstream/main` at the baseline commit:

- tracked by the baseline and unchanged: `upstream_unchanged`;
- tracked by the baseline and changed: `upstream_modified_locally`;
- absent from the baseline: `local_addition`;
- known third-party path or source notice: `third_party_vendored_or_adapted`.

The table is an engineering provenance record, not a legal determination or plagiarism
certificate. Its purpose is to make the repository's derivation explicit and auditable.
