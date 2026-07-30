# ARC (Automatic Recovery Controller) - Self-Healing Neural Networks
# Copyright (c) 2026 Aryan Kaushik. All rights reserved.
#
# This file is part of ARC.
#
# ARC is free software: you can redistribute it and/or modify it under the
# terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.
#
# ARC is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for
# more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ARC. If not, see <https://www.gnu.org/licenses/>.

import builtins
import importlib
import os
import sys

import numpy as np
import pytest


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _block_scipy_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "scipy" or name.startswith("scipy."):
            raise ModuleNotFoundError("No module named 'scipy'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def _purge_arc_modules() -> None:
    for module_name in list(sys.modules.keys()):
        if module_name == "arc" or module_name.startswith("arc."):
            sys.modules.pop(module_name, None)


def test_import_arc_succeeds_without_scipy(monkeypatch: pytest.MonkeyPatch) -> None:
    _block_scipy_imports(monkeypatch)
    _purge_arc_modules()

    arc = importlib.import_module("arc")
    assert arc is not None

    _purge_arc_modules()


@pytest.mark.parametrize(
    "callable_factory, expected_feature_name",
    [
        (
            lambda fe: lambda: fe._compute_spectral_features(np.array([1.0, 2.0, 3.0, 4.0]), "sig"),
            "spectral",
        ),
        (
            lambda fe: lambda: fe._compute_anomaly_features(np.array([1.0, 2.0, 3.0, 4.0]), "sig"),
            "anomaly",
        ),
        (
            lambda fe: lambda: fe._compute_correlation_features(
                {
                    "gradient.global.total_grad_norm_l2": np.array([1.0, 2.0, 3.0]),
                    "loss.trajectory.loss_gradient": np.array([1.0, 2.0, 3.0]),
                }
            ),
            "correlation",
        ),
    ],
)
def test_scipy_dependent_paths_raise_helpful_importerror(
    monkeypatch: pytest.MonkeyPatch,
    callable_factory,
    expected_feature_name: str,
) -> None:
    _block_scipy_imports(monkeypatch)

    from arc.features.extractor import FeatureExtractor

    extractor = FeatureExtractor()

    with pytest.raises(ImportError) as excinfo:
        callable_factory(extractor)()  # call the generated callable

    msg = str(excinfo.value)
    assert f"SciPy is required for {expected_feature_name} features" in msg
    assert "pip install arc-training[full]" in msg


def test_non_scipy_feature_extraction_still_works(monkeypatch: pytest.MonkeyPatch) -> None:
    _block_scipy_imports(monkeypatch)

    from arc.features.extractor import FeatureExtractor

    extractor = FeatureExtractor(compute_trend=False, compute_spectral=False, compute_correlations=False)

    features = extractor.extract_features(np.array([1.0, 2.0]), "sig")
    assert features["sig_mean"] == pytest.approx(1.5)
    assert "sig_std" in features
