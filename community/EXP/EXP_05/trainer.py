"""
EXP_05 — Contrastive Sparse Autoencoder Suppression (C-SAES)
=============================================================
Lý thuyết (PreRound_1.md — Alternative 2: C-SAES, RANK #1):

  Thay vì can thiệp vào weight space (dễ gây representation collapse),
  C-SAES hoạt động trong activation space của mô hình bằng cách:

  1. FEATURE IDENTIFICATION (offline, trước training):
     - Chạy forget corpus và retain corpus qua model → lấy residual stream
       activations tại một layer target (thường mid-to-late layer)
     - Tính mean activation vector trên forget vs retain
     - "Forget Direction" = mean_forget - mean_retain (normalized)
       → đây là direction trong activation space đặc trưng cho forget data

  2. CONTRASTIVE SUPPRESSION (online, trong training):
     Thay vì dùng SAE thật (cần pre-trained SAE model riêng, không có
     sẵn cho Llama-3.2-1B), ta approximate bằng:
       - Projection loss: suppress activation component dọc theo forget_direction
         khi gặp forget inputs
       - Contrastive loss: duy trì activation component trên retain direction
         khi gặp retain inputs
     Loss = γ · L_suppression(forget) + α · L_retain(NLL) + β · L_contrastive

  Tại sao tránh collapse?
    - Can thiệp ở activation space, không phải weight space
    - Chỉ suppress "forget direction" MỘT vector cụ thể, không touch các
      directions khác → highly surgical
    - Contrastive term đảm bảo retain activations không bị drift

  Ưu điểm so với ER-BWP:
    - Scalable infinitely cho sequential (mỗi shard chỉ cần compute
      một forget_direction mới, independent với các shard trước)
    - FIM không cần recompute → không có non-geodesic drift
    - Maths: L_supp = ||A_f · d_forget||² → chỉ minimize projection

  Reference: PreRound_1.md § "Alternative 2: Contrastive Sparse Autoencoder
             Suppression (C-SAES)" — ranked #1 for sequential unlearning
"""

import copy
import logging

import torch
import torch.nn.functional as F
from transformers.integrations.deepspeed import is_deepspeed_available

if is_deepspeed_available():
    import deepspeed

from trainer.unlearn.grad_diff import GradDiff

logger = logging.getLogger(__name__)


class CSAES(GradDiff):
    """
    Contrastive Sparse Autoencoder Suppression (C-SAES).

    Approximates the SAE-based approach from PreRound_1 for the TOFU setting
    where no pre-trained SAE is available: uses contrastive mean-activation
    directions as the "sparse feature directions" to suppress.

    Args:
        target_layer  : Which transformer layer index to hook for activations.
                        -1 = last layer, -2 = second to last, etc.
                        Default: -8 (roughly 60% depth for 1B models)
        proj_coeff    : Weight of the suppression loss on forget activations.
        contrast_coeff: Weight of the contrastive preservation loss on retain.
        warmup_steps  : Steps before suppression is active (compute direction first).
        alpha         : Retain NLL weight (from GradDiff).
        gamma         : Forget loss weight (from GradDiff).
    """

    def __init__(
        self,
        target_layer: int = -8,
        proj_coeff: float = 1.0,
        contrast_coeff: float = 0.5,
        warmup_steps: int = 10,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.target_layer = target_layer
        self.proj_coeff = proj_coeff
        self.contrast_coeff = contrast_coeff
        self.warmup_steps = warmup_steps

        # Forget direction in activation space (computed after warmup)
        self.forget_direction: torch.Tensor | None = None
        # Accumulation buffers for direction estimation
        self._forget_act_sum: torch.Tensor | None = None
        self._retain_act_sum: torch.Tensor | None = None
        self._forget_act_count: int = 0
        self._retain_act_count: int = 0

        # Hook handle
        self._hook_handle = None
        self._hook_output: torch.Tensor | None = None

    # ──────────────────────────────────────────────────────────
    # Hook management
    # ──────────────────────────────────────────────────────────
    def _get_target_layer(self, model):
        """Get the transformer layer module to attach the hook to."""
        # Unwrap deepspeed / DDP if needed
        base = model
        if is_deepspeed_available() and isinstance(model, deepspeed.DeepSpeedEngine):
            base = model.module
        # Handle accelerate wrapping
        if hasattr(base, "module"):
            base = base.module

        # Navigate to layers
        if hasattr(base, "model") and hasattr(base.model, "layers"):
            layers = base.model.layers
        elif hasattr(base, "transformer") and hasattr(base.transformer, "h"):
            layers = base.transformer.h
        else:
            # Fallback: try to find any list of layers
            for attr in ["layers", "encoder", "decoder"]:
                if hasattr(base, attr):
                    candidate = getattr(base, attr)
                    if hasattr(candidate, "__len__"):
                        layers = candidate
                        break
            else:
                return None
        return layers[self.target_layer]

    def _register_hook(self, model):
        layer = self._get_target_layer(model)
        if layer is None:
            return

        def hook_fn(module, input, output):
            # output is typically (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            # Mean over sequence length → [B, D]
            self._hook_output = hidden.mean(dim=1).detach()

        self._hook_handle = layer.register_forward_hook(hook_fn)

    def _remove_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    # ──────────────────────────────────────────────────────────
    # Direction computation
    # ──────────────────────────────────────────────────────────
    def _accumulate_direction(self, activation: torch.Tensor, is_forget: bool):
        """Accumulate mean activations to compute forget/retain directions."""
        act = activation.float()  # [B, D]
        if is_forget:
            if self._forget_act_sum is None:
                self._forget_act_sum = act.sum(0)
            else:
                self._forget_act_sum += act.sum(0)
            self._forget_act_count += act.shape[0]
        else:
            if self._retain_act_sum is None:
                self._retain_act_sum = act.sum(0)
            else:
                self._retain_act_sum += act.sum(0)
            self._retain_act_count += act.shape[0]

    def _finalize_direction(self):
        """Compute and normalize the forget direction from accumulated means."""
        if (
            self._forget_act_sum is None
            or self._retain_act_sum is None
            or self._forget_act_count == 0
            or self._retain_act_count == 0
        ):
            return
        mean_forget = self._forget_act_sum / self._forget_act_count
        mean_retain = self._retain_act_sum / self._retain_act_count
        direction = mean_forget - mean_retain
        norm = direction.norm()
        if norm > 1e-8:
            direction = direction / norm
        self.forget_direction = direction
        logger.info(
            f"[C-SAES] Forget direction computed at step {self.state.global_step}. "
            f"||direction||={norm.item():.4f}"
        )

    # ──────────────────────────────────────────────────────────
    # Suppression + contrastive losses
    # ──────────────────────────────────────────────────────────
    def _suppression_loss(self, activations: torch.Tensor) -> torch.Tensor:
        """
        L_supp = mean(||A · d||²) where d = forget_direction.
        Minimizing this suppresses the forget-direction component of activations.
        """
        d = self.forget_direction.to(activations.device)
        proj = (activations.float() @ d).pow(2)  # [B]
        return proj.mean()

    def _contrastive_loss(self, activations: torch.Tensor) -> torch.Tensor:
        """
        L_contrast = -mean(||A · d||) for retain activations.
        Maximizing the projection on retain set ensures d stays aligned
        with forget-specific features, not general features.
        """
        d = self.forget_direction.to(activations.device)
        proj = (activations.float() @ d).abs()  # [B]
        # We WANT this to stay high for retain (minimize -projection)
        return -proj.mean()

    # ──────────────────────────────────────────────────────────
    # Main compute_loss
    # ──────────────────────────────────────────────────────────
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        step = self.state.global_step if self.state is not None else 0
        forget_inputs = inputs["forget"]
        retain_inputs_raw = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs_raw["input_ids"],
            "attention_mask": retain_inputs_raw["attention_mask"],
            "labels": retain_inputs_raw["labels"],
        }

        # ── 1. Retain NLL loss (always) ────────────────────────────
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        # ── 2. Forget loss — standard GradAscent (negative NLL) ────
        forget_clean = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }
        forget_outputs = model(**forget_clean)
        forget_loss = -forget_outputs.loss  # gradient ascent

        loss = self.gamma * forget_loss + self.alpha * retain_loss

        # ── 3. Direction accumulation during warmup ─────────────────
        if step < self.warmup_steps:
            self._register_hook(model)
            # Forward for forget activations
            with torch.no_grad():
                model(**forget_clean)
            if self._hook_output is not None:
                self._accumulate_direction(self._hook_output, is_forget=True)

            # Forward for retain activations
            with torch.no_grad():
                model(**retain_inputs)
            if self._hook_output is not None:
                self._accumulate_direction(self._hook_output, is_forget=False)

            self._remove_hook()

            # Finalize direction at end of warmup
            if step == self.warmup_steps - 1:
                self._finalize_direction()

        # ── 4. Suppression + contrastive losses (post-warmup) ───────
        elif self.forget_direction is not None:
            self._register_hook(model)

            # Forward forget to get activation (with grad for suppression)
            _ = model(**forget_clean)
            if self._hook_output is not None:
                supp_loss = self._suppression_loss(self._hook_output)
                loss = loss + self.proj_coeff * supp_loss
            else:
                supp_loss = torch.tensor(0.0)

            # Forward retain to get activation (no grad needed)
            with torch.no_grad():
                model(**retain_inputs)
            if self._hook_output is not None:
                cont_loss = self._contrastive_loss(self._hook_output)
                loss = loss + self.contrast_coeff * cont_loss
            else:
                cont_loss = torch.tensor(0.0)

            self._remove_hook()

            if step % 10 == 0:
                logger.info(
                    f"[C-SAES] step={step:4d} | "
                    f"forget={forget_loss.item():.4f} | "
                    f"retain={retain_loss.item():.4f} | "
                    f"suppress={supp_loss.item():.4f} | "
                    f"contrastive={cont_loss.item():.4f} | "
                    f"total={loss.item():.4f}"
                )

        return (loss, forget_outputs) if return_outputs else loss
