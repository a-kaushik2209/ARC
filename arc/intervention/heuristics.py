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

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque
import math

@dataclass
class DiffusionConfig:
    track_attention_entropy: bool = True
    min_attention_entropy: float = 0.1
    max_attention_entropy: float = 5.0

    track_skip_connections: bool = True
    skip_magnitude_threshold: float = 100.0

    critical_timesteps: List[int] = field(default_factory=lambda: [0, 100, 500, 999])
    track_per_timestep: bool = True

class DiffusionHeuristics:
    def __init__(
        self,
        model: nn.Module,
        config: Optional[DiffusionConfig] = None,
        verbose: bool = True,
    ):
        self.model = model
        self.config = config or DiffusionConfig()
        self.verbose = verbose

        self._attention_layers = self._find_attention_layers()
        self._attention_hooks = []
        self._attention_outputs = {}

        self._entropy_history = deque(maxlen=100)
        self._skip_history = deque(maxlen=100)

        if self.config.track_attention_entropy:
            self._register_attention_hooks()

        if verbose:
            print(f"  DiffusionHeuristics: Found {len(self._attention_layers)} attention layers")

    def _find_attention_layers(self) -> List[Tuple[str, nn.Module]]:
        attention_layers = []
        for name, module in self.model.named_modules():
            if any(attn in name.lower() for attn in ['attn', 'attention', 'crossattn']):
                if isinstance(module, (nn.MultiheadAttention,)) or hasattr(module, 'to_q'):
                    attention_layers.append((name, module))
        return attention_layers

    def _register_attention_hooks(self):
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                self._attention_outputs[name] = output.detach()
            return hook

        for name, layer in self._attention_layers[:5]:
            handle = layer.register_forward_hook(make_hook(name))
            self._attention_hooks.append(handle)

    def _compute_attention_entropy(self) -> float:
        if not self._attention_outputs:
            return -1.0

        entropies = []
        for name, attn_out in self._attention_outputs.items():
            attn_probs = torch.softmax(attn_out.flatten(), dim=-1)
            entropy = -torch.sum(attn_probs * torch.log(attn_probs + 1e-10))
            entropies.append(entropy.item())

        return sum(entropies) / len(entropies) if entropies else -1.0

    def check(
        self,
        output: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        result = {
            'critical': False,
            'warnings': [],
            'metrics': {},
            'type': None,
        }

        if self.config.track_attention_entropy:
            entropy = self._compute_attention_entropy()
            result['metrics']['attention_entropy'] = entropy
            self._entropy_history.append(entropy)

            if entropy >= 0:
                if entropy < self.config.min_attention_entropy:
                    result['critical'] = True
                    result['type'] = 'attention_collapse'
                    result['warnings'].append(f"Attention collapsed (entropy={entropy:.4f})")
                elif entropy > self.config.max_attention_entropy:
                    result['warnings'].append(f"Attention too diffuse (entropy={entropy:.4f})")

        output_max = output.abs().max().item()
        output_mean = output.abs().mean().item()
        result['metrics']['output_max'] = output_max
        result['metrics']['output_mean'] = output_mean

        if output_max > 1e6:
            result['critical'] = True
            result['type'] = 'output_explosion'
            result['warnings'].append(f"Output explosion (max={output_max:.2e})")

        if timesteps is not None and self.config.track_per_timestep:
            t = timesteps.mean().item() if timesteps.numel() > 1 else timesteps.item()
            result['metrics']['timestep'] = t

            if t < 100 and output_mean > 10:
                result['warnings'].append(f"High output at low timestep ({t})")

        self._attention_outputs.clear()

        return result

    def cleanup(self):
        for handle in self._attention_hooks:
            handle.remove()
        self._attention_hooks.clear()

@dataclass
class LLMConfig:
    track_embedding_gradients: bool = True
    embedding_grad_threshold: float = 100.0

    track_attention_patterns: bool = True
    detect_repetition: bool = True
    max_repetition_ratio: float = 0.8

    track_layer_norms: bool = True
    norm_explosion_threshold: float = 1000.0

    track_perplexity: bool = True
    perplexity_spike_threshold: float = 10.0

class LLMHeuristics:
    def __init__(
        self,
        model: nn.Module,
        config: Optional[LLMConfig] = None,
        verbose: bool = True,
    ):
        self.model = model
        self.config = config or LLMConfig()
        self.verbose = verbose

        self._embedding_layer = self._find_embedding_layer()
        self._norm_layers = self._find_norm_layers()

        self._perplexity_history = deque(maxlen=50)
        self._embedding_grad_history = deque(maxlen=50)

        if verbose:
            emb_name = self._embedding_layer[0] if self._embedding_layer else "None"
            print(f"  LLMHeuristics: Embedding layer={emb_name}, Norms={len(self._norm_layers)}")

    def _find_embedding_layer(self) -> Optional[Tuple[str, nn.Module]]:
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Embedding):
                if any(kw in name.lower() for kw in ['embed', 'token', 'wte', 'word']):
                    return (name, module)
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Embedding):
                return (name, module)
        return None

    def _find_norm_layers(self) -> List[Tuple[str, nn.Module]]:
        norms = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.LayerNorm, nn.RMSNorm if hasattr(nn, 'RMSNorm') else nn.LayerNorm)):
                norms.append((name, module))
            elif 'rmsnorm' in type(module).__name__.lower():
                norms.append((name, module))
        return norms

    def check(
        self,
        logits: torch.Tensor,
        loss: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        result = {
            'critical': False,
            'warnings': [],
            'metrics': {},
            'type': None,
        }

        if self.config.track_embedding_gradients and self._embedding_layer:
            _, emb = self._embedding_layer
            if emb.weight.grad is not None:
                emb_grad_norm = emb.weight.grad.norm().item()
                result['metrics']['embedding_grad_norm'] = emb_grad_norm
                self._embedding_grad_history.append(emb_grad_norm)

                if emb_grad_norm > self.config.embedding_grad_threshold:
                    result['critical'] = True
                    result['type'] = 'embedding_gradient_explosion'
                    result['warnings'].append(f"Embedding grad explosion ({emb_grad_norm:.2e})")

        if self.config.detect_repetition:
            with torch.no_grad():
                predictions = logits.argmax(dim=-1)
                if predictions.numel() > 10:
                    last_seq = predictions[0, -20:] if predictions.dim() > 1 else predictions[-20:]
                    unique_ratio = len(torch.unique(last_seq)) / len(last_seq)
                    result['metrics']['unique_token_ratio'] = unique_ratio

                    if unique_ratio < (1 - self.config.max_repetition_ratio):
                        result['warnings'].append(f"Repetitive output detected ({unique_ratio:.2%} unique)")

        if self.config.track_perplexity and loss is not None:
            perplexity = torch.exp(loss).item()
            result['metrics']['perplexity'] = perplexity
            self._perplexity_history.append(perplexity)

            if len(self._perplexity_history) >= 5:
                avg_ppl = sum(list(self._perplexity_history)[-5:-1]) / 4
                if avg_ppl > 0 and perplexity / avg_ppl > self.config.perplexity_spike_threshold:
                    result['critical'] = True
                    result['type'] = 'perplexity_spike'
                    result['warnings'].append(f"Perplexity spike ({perplexity:.2f} vs avg {avg_ppl:.2f})")

        if self.config.track_layer_norms:
            for name, norm in self._norm_layers[:5]:
                if hasattr(norm, 'weight') and norm.weight is not None:
                    norm_val = norm.weight.abs().max().item()
                    if norm_val > self.config.norm_explosion_threshold:
                        result['critical'] = True
                        result['type'] = 'norm_explosion'
                        result['warnings'].append(f"LayerNorm explosion in {name} ({norm_val:.2e})")
                        break

        return result

@dataclass
class DetectionConfig:
    track_component_losses: bool = True
    cls_loss_threshold: float = 50.0
    box_loss_threshold: float = 20.0
    obj_loss_threshold: float = 50.0

    track_box_predictions: bool = True
    max_box_size: float = 2.0
    min_box_size: float = 0.001

    track_nms_outputs: bool = True
    max_detections_per_image: int = 1000

class DetectionHeuristics:
    def __init__(
        self,
        model: nn.Module,
        config: Optional[DetectionConfig] = None,
        verbose: bool = True,
    ):
        self.model = model
        self.config = config or DetectionConfig()
        self.verbose = verbose

        self._loss_history = {
            'cls': deque(maxlen=50),
            'box': deque(maxlen=50),
            'obj': deque(maxlen=50),
        }
        self._detection_counts = deque(maxlen=50)

        if verbose:
            print(f"  DetectionHeuristics initialized")

    def check(
        self,
        predictions: torch.Tensor,
        losses: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        result = {
            'critical': False,
            'warnings': [],
            'metrics': {},
            'type': None,
        }

        if self.config.track_component_losses and losses:
            for loss_type, threshold in [
                ('cls', self.config.cls_loss_threshold),
                ('box', self.config.box_loss_threshold),
                ('obj', self.config.obj_loss_threshold),
            ]:
                if loss_type in losses:
                    loss_val = losses[loss_type].item() if isinstance(losses[loss_type], torch.Tensor) else losses[loss_type]
                    result['metrics'][f'{loss_type}_loss'] = loss_val
                    self._loss_history[loss_type].append(loss_val)

                    if loss_val > threshold:
                        result['critical'] = True
                        result['type'] = f'{loss_type}_loss_explosion'
                        result['warnings'].append(f"{loss_type} loss explosion ({loss_val:.2f})")

            if all(len(self._loss_history[k]) > 0 for k in ['cls', 'box']):
                cls_avg = sum(self._loss_history['cls']) / len(self._loss_history['cls'])
                box_avg = sum(self._loss_history['box']) / len(self._loss_history['box'])
                if cls_avg > 0 and box_avg / cls_avg > 100:
                    result['warnings'].append(f"Loss imbalance: box/cls ratio = {box_avg/cls_avg:.1f}")

        if self.config.track_box_predictions and predictions is not None:
            if predictions.dim() >= 2 and predictions.size(-1) >= 4:
                boxes = predictions[..., :4]

                box_max = boxes.abs().max().item()
                box_min = boxes.abs().min().item()

                result['metrics']['max_box_coord'] = box_max

                if box_max > self.config.max_box_size:
                    result['warnings'].append(f"Large box coordinates ({box_max:.2f})")

                if boxes.dim() >= 2:
                    widths = boxes[..., 2] - boxes[..., 0]
                    heights = boxes[..., 3] - boxes[..., 1]
                    if (widths < 0).any() or (heights < 0).any():
                        result['critical'] = True
                        result['type'] = 'invalid_boxes'
                        result['warnings'].append("Negative box dimensions detected")

        return result

class AutoHeuristics:

    def __init__(self, model: nn.Module, verbose: bool = True):
        self.model = model
        self.verbose = verbose

        self.model_type = self._detect_model_type()

        if self.model_type == 'diffusion':
            self.heuristics = DiffusionHeuristics(model, verbose=verbose)
        elif self.model_type == 'llm':
            self.heuristics = LLMHeuristics(model, verbose=verbose)
        elif self.model_type == 'detection':
            self.heuristics = DetectionHeuristics(model, verbose=verbose)
        else:
            self.heuristics = None
            if verbose:
                print(f"  AutoHeuristics: Unknown model type, using generic monitoring")

    def _detect_model_type(self) -> str:
        model_name = type(self.model).__name__.lower()
        module_names = [name.lower() for name, _ in self.model.named_modules()]
        module_str = ' '.join(module_names)

        if any(kw in model_name for kw in ['unet', 'diffusion', 'ddpm', 'ldm']):
            return 'diffusion'
        if 'timestep' in module_str or ('down_block' in module_str and 'up_block' in module_str):
            return 'diffusion'

        if any(kw in model_name for kw in ['gpt', 'llama', 'bert', 'transformer', 'lm']):
            return 'llm'
        if 'embed_tokens' in module_str or 'lm_head' in module_str:
            return 'llm'

        if any(kw in model_name for kw in ['yolo', 'rcnn', 'retinanet', 'fcos', 'ssd']):
            return 'detection'
        if 'backbone' in module_str and ('neck' in module_str or 'head' in module_str):
            return 'detection'

        return 'unknown'

    def check(self, output: torch.Tensor, **kwargs) -> Dict[str, Any]:
        if self.heuristics is None:
            return {
                'critical': False,
                'warnings': [],
                'metrics': {'model_type': 'unknown'},
                'type': None,
            }

        return self.heuristics.check(output, **kwargs)

if __name__ == "__main__":
    print("Testing Model-Specific Heuristics...")

    class FakeLLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(1000, 256)
            self.layers = nn.ModuleList([nn.Linear(256, 256) for _ in range(4)])
            self.norm = nn.LayerNorm(256)
            self.lm_head = nn.Linear(256, 1000)

        def forward(self, x):
            x = self.embed_tokens(x)
            for layer in self.layers:
                x = layer(x)
            x = self.norm(x)
            return self.lm_head(x)

    model = FakeLLM()
    auto = AutoHeuristics(model)
    print(f"  Detected model type: {auto.model_type}")

    x = torch.randint(0, 1000, (2, 32))
    out = model(x)
    loss = out.mean()
    loss.backward()

    result = auto.check(out, loss=loss)
    print(f"  Check result: critical={result['critical']}, warnings={len(result['warnings'])}")