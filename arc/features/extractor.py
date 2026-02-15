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

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from scipy import stats
from scipy.fft import fft

class FeatureExtractor:
    def __init__(
        self,
        window_size: int = 10,
        compute_trend: bool = True,
        compute_spectral: bool = False,
        compute_correlations: bool = True,
    ):
        self.window_size = window_size
        self.compute_trend = compute_trend
        self.compute_spectral = compute_spectral
        self.compute_correlations = compute_correlations

        self.correlation_pairs = [
            ("gradient.global.total_grad_norm_l2", "loss.trajectory.loss_gradient"),
            ("weight.global.mean_update_ratio", "loss.epoch.train_loss"),
            ("activation.global.mean_similarity", "gradient.global.grad_flow_ratio"),
            ("optimizer.global.effective_lr", "weight.global.mean_update_norm"),
        ]

    def extract_features(
        self,
        signal_history: np.ndarray,
        signal_name: str
    ) -> Dict[str, float]:
        features = {}

        if len(signal_history) == 0:
            return features

        window = signal_history[-self.window_size:]
        n = len(window)

        if n == 0:
            return features

        features[f"{signal_name}_mean"] = np.mean(window)
        features[f"{signal_name}_std"] = np.std(window) if n > 1 else 0.0
        features[f"{signal_name}_min"] = np.min(window)
        features[f"{signal_name}_max"] = np.max(window)
        features[f"{signal_name}_range"] = features[f"{signal_name}_max"] - features[f"{signal_name}_min"]

        if n >= 2:
            features[f"{signal_name}_median"] = np.median(window)

            q1, q3 = np.percentile(window, [25, 75])
            features[f"{signal_name}_iqr"] = q3 - q1

        features[f"{signal_name}_current"] = signal_history[-1]
        if features[f"{signal_name}_range"] > 1e-10:
            features[f"{signal_name}_relative_pos"] = (
                (features[f"{signal_name}_current"] - features[f"{signal_name}_min"]) /
                features[f"{signal_name}_range"]
            )

        if n >= 2:
            diff = np.diff(window)
            features[f"{signal_name}_diff_mean"] = np.mean(diff)
            features[f"{signal_name}_diff_std"] = np.std(diff)
            features[f"{signal_name}_diff_last"] = diff[-1] if len(diff) > 0 else 0.0

            if n >= 3:
                diff2 = np.diff(diff)
                features[f"{signal_name}_diff2_mean"] = np.mean(diff2)
                features[f"{signal_name}_diff2_last"] = diff2[-1] if len(diff2) > 0 else 0.0

        if self.compute_trend and n >= 3:
            trend_features = self._compute_trend_features(window, signal_name)
            features.update(trend_features)

        if self.compute_spectral and n >= 4:
            spectral_features = self._compute_spectral_features(window, signal_name)
            features.update(spectral_features)

        if n >= 3:
            anomaly_features = self._compute_anomaly_features(window, signal_name)
            features.update(anomaly_features)

        return features

    def extract_all_features(
        self,
        signal_histories: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        all_features = {}

        for signal_name, history in signal_histories.items():
            if len(history) == 0:
                continue
            features = self.extract_features(history, signal_name)
            all_features.update(features)

        if self.compute_correlations:
            corr_features = self._compute_correlation_features(signal_histories)
            all_features.update(corr_features)

        return all_features

    def _compute_trend_features(
        self,
        window: np.ndarray,
        signal_name: str
    ) -> Dict[str, float]:
        features = {}
        n = len(window)

        if n < 3:
            return features

        x = np.arange(n)

        try:
            slope, intercept = np.polyfit(x, window, 1)
            features[f"{signal_name}_trend_slope"] = slope
            features[f"{signal_name}_trend_intercept"] = intercept

            mean = np.mean(window)
            if abs(mean) > 1e-10:
                features[f"{signal_name}_trend_slope_norm"] = slope / mean

            predicted = slope * x + intercept
            residuals = window - predicted
            features[f"{signal_name}_trend_residual_std"] = np.std(residuals)

            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((window - mean) ** 2)
            if ss_tot > 1e-10:
                features[f"{signal_name}_trend_r2"] = 1 - (ss_res / ss_tot)

            if n >= 4:
                coeffs = np.polyfit(x, window, 2)
                features[f"{signal_name}_curvature"] = coeffs[0]

        except np.linalg.LinAlgError:
            pass

        return features

    def _compute_spectral_features(
        self,
        window: np.ndarray,
        signal_name: str
    ) -> Dict[str, float]:
        features = {}
        n = len(window)

        if n < 4:
            return features

        try:
            centered = window - np.mean(window)

            fft_vals = fft(centered)
            power = np.abs(fft_vals[:n//2]) ** 2
            if len(power) > 1:
                total_power = np.sum(power)
                features[f"{signal_name}_spectral_power"] = total_power

                if total_power > 1e-10:
                    dom_freq_idx = np.argmax(power[1:]) + 1
                    features[f"{signal_name}_dominant_freq"] = dom_freq_idx / n
                    features[f"{signal_name}_dominant_power_ratio"] = power[dom_freq_idx] / total_power

                power_norm = power / (total_power + 1e-10)
                power_norm = power_norm[power_norm > 1e-10]
                if len(power_norm) > 0:
                    spectral_entropy = -np.sum(power_norm * np.log(power_norm))
                    features[f"{signal_name}_spectral_entropy"] = spectral_entropy

        except Exception:
            pass

        return features

    def _compute_anomaly_features(
        self,
        window: np.ndarray,
        signal_name: str
    ) -> Dict[str, float]:
        features = {}
        n = len(window)

        if n < 3:
            return features

        mean = np.mean(window)
        std = np.std(window)

        if std > 1e-10:
            features[f"{signal_name}_zscore"] = (window[-1] - mean) / std

            zscores = (window - mean) / std
            features[f"{signal_name}_max_zscore"] = np.max(np.abs(zscores))

        if n >= 4:
            try:
                features[f"{signal_name}_skewness"] = stats.skew(window)
                features[f"{signal_name}_kurtosis"] = stats.kurtosis(window)
            except Exception:
                pass

        if n >= 2:
            diffs = np.diff(window)
            sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
            features[f"{signal_name}_oscillation_count"] = sign_changes

        return features

    def _compute_correlation_features(
        self,
        signal_histories: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        features = {}

        for signal_a, signal_b in self.correlation_pairs:
            if signal_a not in signal_histories or signal_b not in signal_histories:
                continue

            hist_a = signal_histories[signal_a][-self.window_size:]
            hist_b = signal_histories[signal_b][-self.window_size:]

            min_len = min(len(hist_a), len(hist_b))
            if min_len < 3:
                continue

            hist_a = hist_a[-min_len:]
            hist_b = hist_b[-min_len:]

            try:
                corr, pvalue = stats.pearsonr(hist_a, hist_b)

                safe_name = f"{signal_a.split('.')[-1]}_{signal_b.split('.')[-1]}"
                features[f"corr_{safe_name}"] = corr
                features[f"corr_{safe_name}_pvalue"] = pvalue

            except Exception:
                continue

        return features

    def get_feature_names(self, signal_names: List[str]) -> List[str]:
        dummy_history = np.random.randn(self.window_size)
        all_names = []

        for signal_name in signal_names:
            features = self.extract_features(dummy_history, signal_name)
            all_names.extend(features.keys())

        return sorted(set(all_names))