# Contributing to SoftSkill

Thank you for your interest in contributing. SoftSkill welcomes focused fixes, documentation improvements, tests, benchmark integrations, and soft-prefix training improvements.

## Getting Started

```bash
git clone https://github.com/SanchesLearnAI/Softskill.git
cd Softskill
pip install -e ".[dev,softprefix]"
```

## How To Contribute

### Bug Reports

Open a GitHub issue with reproduction steps, expected behavior, actual behavior, and the config file you used. Remove API keys, local paths, and private data.

### Add A Benchmark

See the benchmark guide in `docs/guide/new-benchmark.md` and the scaffold at `skillopt/envs/_template/`.

### Improve Soft-Prefix Training

Keep changes covered by tests where practical. Avoid committing generated `outputs/`, `rollouts/`, local corpora, or model checkpoints.

### Improve Documentation

```bash
pip install -e ".[docs]"
mkdocs serve
```

## Pull Request Process

1. Fork the repo and create a feature branch.
2. Make focused changes and add or update tests.
3. Run the relevant test subset locally.
4. Submit a PR with a clear description and note any skipped verification.

## Code Style

- Follow existing patterns in the codebase.
- Use type hints for new public functions.
- Keep docstrings concise.
- Keep release-facing scripts parameterized through environment variables.

## License

By contributing, you agree your contributions are licensed under the MIT License.
Do not remove upstream attribution, `PROVENANCE.md`, `CODE_PROVENANCE.csv`, or
third-party notices. Describe whether each contribution modifies an upstream file or
adds a new local research component.
