from __future__ import annotations
import numpy as np


def dpo_loss(
    policy_chosen_logps: np.ndarray,
    policy_rejected_logps: np.ndarray,
    ref_chosen_logps: np.ndarray,
    ref_rejected_logps: np.ndarray,
    beta: float,
) -> float:
    """Compute batch DPO loss from sequence log probabilities.

    DPO objective (Rafailov et al. 2023):
        loss = -E[ log σ( β * (log π_θ(y_w|x) - log π_ref(y_w|x))
                           - β * (log π_θ(y_l|x) - log π_ref(y_l|x)) ) ]

    Numerically stable: use log-sigmoid via -log1p(exp(-z)).
    """
    # Policy log-ratio minus reference log-ratio
    chosen_logratios = policy_chosen_logps - ref_chosen_logps
    rejected_logratios = policy_rejected_logps - ref_rejected_logps

    # Scale difference by beta
    z = beta * (chosen_logratios - rejected_logratios)

    # Numerically stable log-sigmoid: log σ(z) = -log(1 + exp(-z))
    # Clamp z to avoid overflow in exp
    z_clamped = np.clip(z, -30.0, 30.0)
    log_sigmoid = -np.log1p(np.exp(-z_clamped))

    loss = -float(np.mean(log_sigmoid))
    return loss


def orpo_loss(
    sft_nll: np.ndarray,
    chosen_logps: np.ndarray,
    rejected_logps: np.ndarray,
    lambda_orpo: float,
) -> float:
    """Compute a simplified ORPO-style objective.

    ORPO (Hong et al. 2024):
        odds(y|x)    = p(y|x) / (1 - p(y|x))
        OR           = odds(y_w|x) / odds(y_l|x)
        preference   = -log σ(log OR)
        total loss   = NLL(y_w) + λ * preference
    """
    # Convert log-probs to probabilities, clip for numerical safety
    chosen_probs = np.clip(np.exp(chosen_logps), 1e-7, 1 - 1e-7)
    rejected_probs = np.clip(np.exp(rejected_logps), 1e-7, 1 - 1e-7)

    # Odds = p / (1 - p)
    chosen_odds = chosen_probs / (1.0 - chosen_probs)
    rejected_odds = rejected_probs / (1.0 - rejected_probs)

    # Log odds ratio
    log_or = np.log(chosen_odds + 1e-7) - np.log(rejected_odds + 1e-7)

    # Preference loss: -log σ(log_or)
    log_or_clamped = np.clip(log_or, -30.0, 30.0)
    pref_loss = -np.mean(np.log(1.0 / (1.0 + np.exp(-log_or_clamped))))

    # SFT loss (mean NLL)
    sft_loss = float(np.mean(sft_nll))

    total: float = sft_loss + lambda_orpo * float(pref_loss)
    return total
