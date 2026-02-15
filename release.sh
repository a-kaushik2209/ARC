#!/bin/bash
# release.sh - One-command release script for ARC

VERSION=$1

if [ -z "$VERSION" ]; then
    echo "Usage: ./release.sh <version>"
    echo "Example: ./release.sh 4.0.0"
    exit 1
fi

# Ensure we are on main
git checkout main
git pull origin main

# Run tests first
echo "Running tests..."
pytest tests/
if [ $? -ne 0 ]; then
    echo "Tests failed! Aborting release."
    exit 1
fi

# Tag release
echo "Tagging v$VERSION..."
git tag -a "v$VERSION" -m "Release v$VERSION"
git push origin "v$VERSION"

# Build package
echo "Building package..."
rm -rf dist/ build/ *.egg-info
python -m build

# Verification
echo "Verifying package..."
twine check dist/*
if [ $? -ne 0 ]; then
    echo "Package verification failed! Aborting release."
    exit 1
fi

# Upload
echo "Uploading to PyPI..."
twine upload dist/*

echo "Successfully released ARC v$VERSION to PyPI!"
