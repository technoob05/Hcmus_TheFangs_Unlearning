"""
EXP_04 — Entropy-Reservoir Bregman-Wasserstein Projection (ER-BWP)
====================================================================
Lý thuyết (PreRound_1.md — Alternative 1: ER-BWP):

  Trong sequential unlearning, mỗi bước update làm thay đổi probability
  measure của latent space. FIM cũ trở nên "stale" → non-geodesic drift
  tích lũy → representation collapse.

  ER-BWP giải quyết bằng cách inject "Entropy Reservoir" vào optimization:
  thay vì chỉ minimize BW divergence đến prior model trên retain set, ER-BWP
  còn duy trì một maximum-entropy coupling trên forget set. Điều này toán học
  bảo đảm: dù forget data bị xóa, semantic capacity tổng thể không collapse
  vào low-dimensional manifold.

  Công thức tổng quát:
    L = γ · L_NPO(forget) + α · L_retain(NLL) + λ(t) · L_entropy(forget)

  Trong đó:
    L_NPO(forget):  NPO loss — KL-guided ascent, geometry-aware hơn GradAscent
    L_retain(NLL):  NLL trên retain set — utility preservation
    L_entropy(forget): -H(p_θ(·|forget)) = entropy maximization trên forget
                        tokens → push model toward "maximal entropy prior"
    λ(t):           dynamic mixing coefficient — decays từ λ_max → λ_min
                    theo training step, simulate entropy reservoir coupling

  Tại sao entropy maximization?
    - Khi model "forgets", ideal state là output distribution trên forget data
      trở về uniform (not knowing) thay vì map sang sai answer.
    - Bằng cách maximize entropy, ta "fill" latent space bằng high-entropy
      distribution thay vì collapse về degenerate distribution.
    - λ(t) decay đảm bảo entropy regularization mạnh ở đầu (prevent collapse)
      và yếu dần sau khi model đã stable (allow targeted forgetting to deepen).

Reference: PreRound_1.md § "Alternative 1: Entropy-Reservoir Bregman-Wasserstein
           Projections (ER-BWP)"
"""

import logging
import math

import torch
import torch.nn.functional as F

from trainer.utils import compute_dpo_loss
from trainer.unlearn.npo import NPO

logger = logging.getLogger(__name__)


class ERBWP(NPO):
    """
    Entropy-Reservoir Bregman-Wasserstein Projection.

    Extends NPO (geometry-aware forget loss) với Entropy Reservoir term:
    maximize entropy của token-level predictions trên forget set bằng một
    dynamic coupling coefficient λ(t) giảm dần theo step.

    Args:
        lambda_max  : Giá trị λ ban đầu (entropy coupling mạnh)
        lambda_min  : Giá trị λ cuối (entropy coupling yếu)
        lambda_decay: "speed" của decay — tỉ lệ số bước để λ đạt lambda_min
                      (dạng cosine schedule từ lambda_max → lambda_min)
        beta        : NPO temperature (kế thừa từ NPO)
        alpha       : Retain loss weight (kế thừa từ GradDiff)
        gamma       : Forget loss weight (kế thừa từ GradDiff)
    """

    def __init__(
        self,
        lambda_max: float = 0.5,
        lambda_min: float = 0.01,
        lambda_decay: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lambda_max = lambda_max
        self.lambda_min = lambda_min
        self.lambda_decay = lambda_decay

    def _get_lambda(self) -> float:
        """
        Dynamic mixing coefficient λ(t) — cosine decay từ lambda_max xuống lambda_min.

        Schedule: λ(t) = λ_min + 0.5*(λ_max - λ_min) * (1 + cos(π * t/T_decay))
        Trong đó T_decay = lambda_decay * max_steps.

        Tại t=0    : λ = λ_max  (entropy reservoir coupling mạnh nhất)
        Tại t=T/2  : λ ≈ (λ_max+λ_min)/2
        Tại t≥T    : λ = λ_min  (entropy reservoir gần như tắt)
        """
        step = self.state.global_step if self.state is not None else 0
        max_steps = max(int(self.args.max_steps), 1)
        t_decay = max(self.lambda_decay * max_steps, 1)
        progress = min(step / t_decay, 1.0)
        lam = self.lambda_min + 0.5 * (self.lambda_max - self.lambda_min) * (
            1.0 + math.cos(math.pi * progress)
        )
        return float(lam)

    def _compute_entropy_loss(self, model, forget_inputs) -> torch.Tensor:
        """
        Entropy Reservoir term: minimize -H(p_θ) trên forget set.

        H = -Σ p_i log p_i  (entropy của token distribution)
        Để MAXIMIZE entropy → minimize (-H) = minimize (Σ p_i log p_i)

        Kết quả: model trở nên "uncertain" về forget data, output probability
        mass spreads uniformly across vocabulary → maximal entropy prior.

        Note: ta dùng logits trực tiếp thay vì cache model(**inputs) riêng
        để tiết kiệm memory — không pass labels vào để tránh compute CE loss.
        """
        outputs = model(
            input_ids=forget_inputs["input_ids"],
            attention_mask=forget_inputs["attention_mask"],
        )
        logits = outputs.logits  # [B, T, V]

        # Per-token softmax probabilities
        probs = F.softmax(logits, dim=-1)  # [B, T, V]

        # Per-token entropy: H(t) = -Σ_v p_v * log(p_v)
        # clamp để tránh log(0)
        token_entropy = -(probs * torch.log(probs.clamp(min=1e-10))).sum(dim=-1)  # [B, T]

        # Mask: chỉ tính trên các token có label hợp lệ (không phải padding)
        labels = forget_inputs.get("labels")
        if labels is not None:
            valid_mask = (labels != -100).float()  # [B, T]
        else:
            valid_mask = forget_inputs["attention_mask"].float()

        # Mean entropy trên valid tokens
        denom = valid_mask.sum().clamp(min=1.0)
        mean_entropy = (token_entropy * valid_mask).sum() / denom

        # minimize -H  =  maximize H  =  push toward uniform distribution
        return -mean_entropy

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        ER-BWP Loss:
            L = γ · L_NPO(forget)    [geometry-aware forget]
              + α · L_retain(NLL)    [utility preservation]
              + λ(t) · L_entropy(forget)  [entropy reservoir coupling]
        """
        forget_inputs = inputs["forget"]
        lam = self._get_lambda()

        # ── 1. NPO forget loss (KL-guided ascent, geometry-aware) ──────
        forget_loss, forget_outputs = compute_dpo_loss(
            model=model,
            ref_model=self.ref_model,
            win_inputs=None,
            lose_inputs=forget_inputs,
            beta=self.beta,
        )

        # ── 2. Retain NLL loss ─────────────────────────────────────────
        retain_inputs = inputs["retain"]
        retain_inputs_clean = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(
            model=model, retain_inputs=retain_inputs_clean
        )

        # ── 3. Entropy Reservoir term ──────────────────────────────────
        forget_inputs_clean = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs.get("labels"),
        }
        entropy_loss = self._compute_entropy_loss(model, forget_inputs_clean)

        # ── 4. Total loss ──────────────────────────────────────────────
        loss = (
            self.gamma * forget_loss
            + self.alpha * retain_loss
            + lam * entropy_loss
        )

        # Log mỗi 10 step để track 3 components riêng
        if self.state is not None and self.state.global_step % 10 == 0:
            logger.info(
                f"[ERBWP] step={self.state.global_step:4d} | "
                f"λ={lam:.4f} | "
                f"forget={forget_loss.item():.4f} | "
                f"retain={retain_loss.item():.4f} | "
                f"entropy={entropy_loss.item():.4f} | "
                f"total={loss.item():.4f}"
            )

        return (loss, forget_outputs) if return_outputs else loss
