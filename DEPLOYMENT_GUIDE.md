# ARC Deployment Guide

## Making ARC Accessible to Everyone

This guide covers all methods (that will be availaible soon) to distribute ARC so anyone can use it with their training.

---

## Quick Start (For Users)

### Method 1: pip install (Recommended)

```bash
# From PyPI (after publishing)
pip install arc-training

# From GitHub directly
pip install git+https://github.com/yourusername/arc-training.git

# Local development install
git clone https://github.com/yourusername/arc-training.git
cd arc-training
pip install -e .
```

### Method 2: One-Line Integration

```python
# Add to ANY PyTorch training script
from arc import WeightRollback

# Wrap your training loop (3 lines!)
arc = WeightRollback(model, optimizer)
for batch in dataloader:
    loss = model(batch)
    action = arc.step(loss)  # Auto-detects and recovers from failures
    if not action.rolled_back:
        loss.backward()
        optimizer.step()
```

---

## Distribution Methods

### 1. PyPI (pip install) - Primary

**Step 1: Update setup.py**

```bash
# Already done - see setup.py updates
```

**Step 2: Create distribution**

```bash
pip install build twine
python -m build
```

**Step 3: Test on TestPyPI**

```bash
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ arc-training
```

**Step 4: Publish to PyPI**

```bash
twine upload dist/*
```

### 2. GitHub Releases

**Step 1: Tag a release**

```bash
git tag -a v4.0.0 -m "ARC v4.0.0 - Production Ready"
git push origin v4.0.0
```

**Step 2: Create GitHub Release**

- Go to GitHub → Releases → New Release
- Select tag v4.0.0
- Upload wheel files from `dist/`
- Add release notes

### 3. Conda-Forge

```yaml
# meta.yaml for conda-forge
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

### 4. Docker Image

```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-runtime

RUN pip install arc-training

# Usage: docker run -it arc-training python train.py
```

---

## Required Files for Distribution

### pyproject.toml (Modern Python packaging)

```toml
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "arc-training"
version = "4.0.0"
description = "Automatic Recovery Controller for Neural Network Training"
readme = "README.md"
license = {text = "MIT"}
authors = [{name = "Aryan Kaushik"}]
requires-python = ">=3.8"
dependencies = [
    "torch>=1.9.0",
    "numpy>=1.19.0",
]

[project.urls]
Homepage = "https://github.com/yourusername/arc-training"
Documentation = "https://arc-training.readthedocs.io"
```

### MANIFEST.in

```
include README.md
include LICENSE
include requirements.txt
recursive-include arc *.py
```

### LICENSE (MIT)

```
MIT License

Copyright (c) 2026 Aryan Kaushik

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software...
```

---

## Documentation Options

### 1. ReadTheDocs (Free)

```bash
pip install sphinx sphinx-rtd-theme
sphinx-quickstart docs
```

### 2. MkDocs (Simple)

```bash
pip install mkdocs mkdocs-material
mkdocs new .
mkdocs serve
```

### 3. GitHub Pages

```bash
mkdocs gh-deploy
```

---

## CI/CD Setup

### GitHub Actions (Auto-publish on release)

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
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
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

## Checklist for v4.0.0 Release

### Pre-release

- [ ] Update version in setup.py/pyproject.toml
- [ ] Update CHANGELOG.md
- [ ] Run full test suite
- [ ] Update README with badges

### Package Files

- [x] setup.py - exists
- [ ] pyproject.toml - create
- [ ] MANIFEST.in - create
- [ ] LICENSE - create/update
- [ ] CHANGELOG.md - create

### Distribution

- [ ] Build wheel: `python -m build`
- [ ] Test locally: `pip install dist/*.whl`
- [ ] Upload to TestPyPI
- [ ] Test from TestPyPI
- [ ] Upload to PyPI
- [ ] Create GitHub release
- [ ] (Optional) Submit to conda-forge

### Documentation

- [ ] Update README.md
- [ ] Create docs with mkdocs/sphinx
- [ ] Deploy to ReadTheDocs/GitHub Pages

---

## One-Command Release

```bash
# Full release script
./release.sh 4.0.0
```

Where release.sh contains:

```bash
#!/bin/bash
VERSION=$1
git tag -a v$VERSION -m "Release v$VERSION"
git push origin v$VERSION
python -m build
twine upload dist/*
echo "Released v$VERSION to PyPI!"
```
