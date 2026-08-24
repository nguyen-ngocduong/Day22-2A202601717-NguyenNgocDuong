import numpy as np

from preference_lab.losses import dpo_loss, orpo_loss


def test_dpo_loss_returns_float() -> None:
    loss = dpo_loss(
        np.array([-0.5]),
        np.array([-1.5]),
        np.array([-0.6]),
        np.array([-1.0]),
        beta=0.1,
    )
    assert isinstance(loss, float)


def test_dpo_loss_preferred_wins() -> None:
    """When policy strongly prefers chosen, loss should be low (close to 0)."""
    # Policy clearly prefers chosen (much higher logp for chosen)
    loss = dpo_loss(
        policy_chosen_logps=np.array([-0.1, -0.2]),
        policy_rejected_logps=np.array([-5.0, -6.0]),
        ref_chosen_logps=np.array([-0.5, -0.5]),
        ref_rejected_logps=np.array([-0.5, -0.5]),
        beta=0.1,
    )
    assert loss < 0.5, f"Expected low DPO loss, got {loss}"


def test_dpo_loss_symmetry() -> None:
    """When policy = reference, loss should be ~log(2) ≈ 0.693."""
    logps = np.array([-1.0])
    loss = dpo_loss(logps, logps, logps, logps, beta=0.1)
    assert abs(loss - np.log(2)) < 1e-5, f"Expected log(2), got {loss}"


def test_orpo_loss_returns_float() -> None:
    loss = orpo_loss(np.array([1.0]), np.array([-0.5]), np.array([-1.5]), lambda_orpo=0.1)
    assert isinstance(loss, float)


def test_orpo_loss_chosen_better() -> None:
    """When chosen logp >> rejected logp, ORPO pref loss should be low."""
    loss = orpo_loss(
        sft_nll=np.array([1.0]),
        chosen_logps=np.array([-0.1]),
        rejected_logps=np.array([-5.0]),
        lambda_orpo=0.1,
    )
    # sft_loss ≈ 1.0, pref_loss should be very small → total ≈ 1.0
    assert loss < 1.5, f"Expected loss close to 1.0, got {loss}"
