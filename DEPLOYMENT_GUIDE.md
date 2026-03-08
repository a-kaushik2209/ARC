<div align="center">

# ARC — Deployment Guide

### Installation, Distribution, and Production Setup

</div>

---

## Quick Start

### Install from PyPI

```bash
pip install arc-training
```

### Install from GitHub

```bash
pip install git+https://github.com/a-kaushik2209/ARC.git
```

### Install for Development

```bash
git clone https://github.com/a-kaushik2209/ARC.git
cd ARC
pip install -e .
```

---

## Integration

### Vanilla PyTorch — 3 Lines

```python
from arc import Arc

controller = Arc(model, optimizer)

for batch in dataloader:
    loss = model(batch)
    action = controller.step(loss)

    if not action.rolled_back:
        loss.backward()
        optimizer.step()
```

### PyTorch Lightning — 1 Line

```python
from arc import ArcCallback

trainer = pl.Trainer(callbacks=[ArcCallback()])
```

### Configuration

```python
from arc import Arc
from arc.config import Config

config = Config()
config.monitoring.check_interval = 50
config.checkpointing.quantized = True
config.recovery.max_rollbacks = 5

controller = Arc(model, optimizer, config=config)
```

---

## Publishing to PyPI

### Build

```bash
pip install build twine
python -m build
```

### Test on TestPyPI

```bash
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ arc-training
```

### Publish

```bash
twine upload dist/*
```

---

## Docker

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-runtime
RUN pip install arc-training
```

```bash
docker build -t arc-training .
docker run -it arc-training python train.py
```

---

## GitHub Releases

```bash
git tag -a v4.0.0 -m "ARC v4.0.0"
git push origin v4.0.0
```

Then create a release on GitHub, attach the wheel files from `dist/`, and add release notes.

---

## CI/CD — Auto-Publish on Release

```yaml
# .github/workflows/publish.yml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - run: pip install build twine
      - run: python -m build
      - run: twine upload dist/*
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
```

---

## Conda-Forge

```yaml
package:
  name: arc-training
  version: 4.0.0

source:
  url: https://pypi.io/packages/source/a/arc-training/arc-training-4.0.0.tar.gz

requirements:
  host:
    - python >=3.8
    - pip
  run:
    - python >=3.8
    - pytorch >=1.9.0
    - numpy >=1.19.0
```

---

## Package Structure

```
arc-training/
├── arc/                    Core library
├── examples/               Integration examples
├── experiments/            Benchmarks and validation scripts
├── setup.py                Package configuration
├── pyproject.toml          Modern Python packaging
├── MANIFEST.in             Distribution manifest
├── LICENSE                 AGPL-3.0
├── README.md               Project documentation
├── CHANGELOG.md            Version history
└── ARC_EFFICIENCY_REPORT.md  Technical report
```

### Required Files

| File             | Purpose                           |
| :--------------- | :-------------------------------- |
| `setup.py`       | Package metadata and dependencies |
| `pyproject.toml` | Modern build system configuration |
| `MANIFEST.in`    | Distribution file inclusion rules |
| `LICENSE`        | AGPL-3.0 license text             |
| `README.md`      | PyPI long description             |

---

## Release Checklist

### Pre-Release

- [ ] Update version in `setup.py` and `pyproject.toml`
- [ ] Update `CHANGELOG.md` with release notes
- [ ] Run full test suite: `python experiments/comprehensive_benchmark.py`
- [ ] Verify all JSON result files are current

### Build and Publish

- [ ] Build: `python -m build`
- [ ] Test locally: `pip install dist/*.whl`
- [ ] Upload to TestPyPI and verify installation
- [ ] Upload to PyPI
- [ ] Create GitHub release with wheel artifacts
- [ ] Tag release: `git tag -a v4.0.0 -m "Release v4.0.0"`

### Post-Release

- [ ] Verify `pip install arc-training` works
- [ ] Update documentation links
- [ ] Announce release

---

<div align="center">

_ARC Deployment Guide · Aryan Kaushik · 2026_

</div>
