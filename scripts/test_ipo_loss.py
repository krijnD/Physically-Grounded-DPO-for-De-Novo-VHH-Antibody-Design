#!/usr/bin/env python3
"""Unit tests for the IPO loss (Brief 18 §5).

Verifies the bounded-regression IPO objective implemented in
:mod:`src.dpo.loss_ipo` matches its mathematical spec on a handful of
constructed PairLosses inputs — no GPU, no diffab forward, no LMDB.

Catches the failure modes most likely to silently survive a smoke run:

  * **Sign error** — if m_per_res is computed with the wrong sign
    (``+T·δ`` instead of ``-T·δ``) the IPO loss still produces a
    plausible positive number, but a winner-preferring solution gets
    pushed *away* from τ. The iter-0 baseline tests would not catch
    this (both signs give the same loss at δ=0); the explicit "winner
    preferred → m=τ → loss=0" test does.

  * **Aggregation collapse** — residue and sequence aggregation give
    different scalars even at δ=0 (residue is N·τ², sequence is τ²
    regardless of N). Both are exercised here.

  * **τ formula error** — τ should be 1/(2β), not 1/β or 2/β. Tested
    explicitly at three β values.

  * **Diagnostics contract** — the trainer's W&B log path expects a
    specific set of keys (loss, margin_mean, tau_target,
    margin_distance_from_tau_mean, converged_pair_fraction, …). Asserts
    they're all present and finite.

Usage::

    python scripts/test_ipo_loss.py
    # or under pytest:
    pytest -q scripts/test_ipo_loss.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "diffab"))

try:
    import torch  # noqa: F401
    from src.dpo.loss import PairLosses
    from src.dpo.loss_ipo import ipo_loss
except (ModuleNotFoundError, ImportError) as _exc:
    print(
        f"SKIP: torch / diffab / src.dpo import failed ({_exc}); "
        f"run on Snellius with the DPO venv active."
    )
    sys.exit(0)


# ── Fixtures ────────────────────────────────────────────────────────


def _zero_pair(B: int, L: int) -> PairLosses:
    """All four channels = 0 for all four positions → δ = 0."""
    def _z():
        return {
            "rot": torch.zeros(B, L),
            "pos": torch.zeros(B, L),
            "seq": torch.zeros(B, L),
        }
    return PairLosses(w_theta=_z(), l_theta=_z(), w_ref=_z(), l_ref=_z())


def _half_mask(B: int, L: int) -> torch.Tensor:
    """[B, L] mask with first L//2 positions active. Mimics generate_flag."""
    m = torch.zeros(B, L)
    m[:, : L // 2] = 1.0
    return m


# ── Iter-0 baseline tests ──────────────────────────────────────────


def test_iter0_residue_aggregation_equals_N_tau_squared() -> None:
    """When π_θ == π_ref the residue-agg IPO loss is N·τ² per pair.

    Each masked residue has m=0; per-residue penalty (0-τ)² = τ²;
    summed over N_masked residues per pair → N·τ²; mean over pairs
    → N·τ² (since all pairs are identical).
    """
    B, L = 4, 20
    pair = _zero_pair(B, L)
    mask = _half_mask(B, L)  # 10 masked residues per pair
    N_masked = int(mask[0].sum().item())
    weights = {"rot": 0.0, "pos": 0.0, "seq": 1.0}  # match floor-IPO seq-only
    for beta in (0.005, 0.05, 0.5):
        loss, _ = ipo_loss(
            pair, mask, weights, beta_dpo=beta, num_timesteps=100,
            aggregation="residue",
        )
        tau = 1.0 / (2.0 * beta)
        expected = N_masked * tau * tau
        got = float(loss.item())
        assert abs(got - expected) < 1e-3, (
            f"β={beta}: expected {expected:.4f}, got {got:.4f} "
            f"(N_masked={N_masked}, τ={tau})"
        )


def test_iter0_sequence_aggregation_equals_tau_squared() -> None:
    """Sequence-agg sums δ first, so iter-0 loss is τ² per pair (NOT N·τ²)."""
    B, L = 4, 20
    pair = _zero_pair(B, L)
    mask = _half_mask(B, L)
    weights = {"rot": 0.0, "pos": 0.0, "seq": 1.0}
    for beta in (0.005, 0.05, 0.5):
        loss, _ = ipo_loss(
            pair, mask, weights, beta_dpo=beta, num_timesteps=100,
            aggregation="sequence",
        )
        tau = 1.0 / (2.0 * beta)
        expected = tau * tau
        got = float(loss.item())
        assert abs(got - expected) < 1e-3, (
            f"β={beta} sequence: expected {expected:.4f}, got {got:.4f}"
        )


# ── Convergence point: loss = 0 when m hits τ exactly ─────────────


def test_zero_loss_when_winner_preferred_to_target_margin() -> None:
    """If L_w_θ improves over L_w_ref by τ/T per residue, m=τ and loss=0.

    Constructs the exact iter-∞ "perfectly preferring winner" state:
    L_w_θ = -τ/T (better than ref by τ/T), L_l_θ = 0 (same as ref),
    L_w_ref = L_l_ref = 0. Then δ = (L_w_θ - L_w_ref) - (L_l_θ - L_l_ref)
    = -τ/T, and m = -T·δ = +τ. Loss collapses to 0 per residue.

    CRITICAL: this test would fail if the m_per_res sign were flipped
    (i.e. if we wrote +T·δ instead of -T·δ). The iter-0 test cannot
    catch that bug because δ=0.
    """
    B, L = 4, 20
    T = 100
    beta = 0.05
    tau = 1.0 / (2.0 * beta)            # 10.0
    mask = _half_mask(B, L)
    weights = {"rot": 0.0, "pos": 0.0, "seq": 1.0}
    # Target: m = -T·δ = τ → δ = -τ/T per masked residue.
    # δ = (L_w_θ - L_w_ref) - (L_l_θ - L_l_ref).
    # Pick L_w_θ = -τ/T, others = 0 → δ = -τ/T. ✓
    seq_w_theta = -(tau / T) * torch.ones(B, L)
    losses_w_theta = {
        "rot": torch.zeros(B, L),
        "pos": torch.zeros(B, L),
        "seq": seq_w_theta,
    }
    zero_set = {
        "rot": torch.zeros(B, L),
        "pos": torch.zeros(B, L),
        "seq": torch.zeros(B, L),
    }
    pair = PairLosses(
        w_theta=losses_w_theta, l_theta=zero_set,
        w_ref=zero_set, l_ref=zero_set,
    )
    loss, diag = ipo_loss(
        pair, mask, weights, beta_dpo=beta, num_timesteps=T,
        aggregation="residue",
    )
    got = float(loss.item())
    assert got < 1e-4, (
        f"Expected near-zero loss when m=τ, got {got:.6f}. "
        f"Possible sign error in m_per_res = -T·δ."
    )
    # Sanity: margin_mean should match τ within float precision.
    margin = float(diag["margin_mean"].item())
    assert abs(margin - tau) < 1e-3, (
        f"Expected margin_mean ≈ τ={tau}, got {margin:.4f}"
    )


# ── Symmetry: bounded objective penalises over- and under-shoot equally ─


def test_symmetry_around_tau() -> None:
    """Loss at m=0 (under-shoot by τ) equals loss at m=2τ (over-shoot by τ).

    This is the defining property of IPO's bounded regression — DPO at
    these two states would give very different log-sigmoid values.
    """
    B, L = 4, 20
    T = 100
    beta = 0.05
    tau = 1.0 / (2.0 * beta)
    mask = _half_mask(B, L)
    weights = {"rot": 0.0, "pos": 0.0, "seq": 1.0}
    # State A: m=0 (iter-0; δ=0)
    pair_A = _zero_pair(B, L)
    loss_A, _ = ipo_loss(pair_A, mask, weights, beta, T, "residue")
    # State B: m=2τ → δ = -2τ/T per residue
    delta_B = -2.0 * tau / T
    seq_w_theta_B = delta_B * torch.ones(B, L)
    zero_set = {
        "rot": torch.zeros(B, L),
        "pos": torch.zeros(B, L),
        "seq": torch.zeros(B, L),
    }
    pair_B = PairLosses(
        w_theta={"rot": torch.zeros(B, L), "pos": torch.zeros(B, L),
                 "seq": seq_w_theta_B},
        l_theta=zero_set, w_ref=zero_set, l_ref=zero_set,
    )
    loss_B, _ = ipo_loss(pair_B, mask, weights, beta, T, "residue")
    assert abs(loss_A.item() - loss_B.item()) < 1e-3, (
        f"Symmetry broken: loss(m=0)={loss_A.item():.4f} vs "
        f"loss(m=2τ)={loss_B.item():.4f}. IPO must be symmetric around τ."
    )


# ── τ formula scaling ──────────────────────────────────────────────


def test_tau_target_diagnostic_matches_formula() -> None:
    """Diagnostic 'tau_target' must equal 1/(2β), not 1/β or 2/β."""
    B, L = 2, 8
    pair = _zero_pair(B, L)
    mask = _half_mask(B, L)
    weights = {"rot": 0.0, "pos": 0.0, "seq": 1.0}
    for beta, expected_tau in [(0.005, 100.0), (0.05, 10.0), (0.5, 1.0)]:
        _, diag = ipo_loss(pair, mask, weights, beta, 100, "residue")
        tau = float(diag["tau_target"].item())
        assert abs(tau - expected_tau) < 1e-6, (
            f"β={beta}: expected τ={expected_tau}, got τ={tau}"
        )


# ── Diagnostics contract (trainer's W&B log path) ─────────────────


def test_diagnostics_contract() -> None:
    """All keys the trainer reads from diag must be present and finite."""
    B, L = 4, 20
    pair = _zero_pair(B, L)
    mask = _half_mask(B, L)
    weights = {"rot": 0.0, "pos": 0.0, "seq": 1.0}
    _, diag = ipo_loss(pair, mask, weights, 0.05, 100, "residue")
    # Keys the trainer's _step explicitly reads:
    required = {
        "loss", "margin_mean", "accuracy",
        "L_w_theta", "L_l_theta", "L_w_ref", "L_l_ref",
        "delta_mean",
        # IPO-specific keys forwarded to W&B (Brief 18 §7):
        "tau_target", "margin_distance_from_tau_mean",
        "converged_pair_fraction",
        "margin_per_residue_mean", "mask_count_mean",
    }
    missing = required - set(diag.keys())
    assert not missing, f"diagnostics missing keys: {missing}"
    for k in required:
        v = diag[k]
        # Every diag value must be a tensor with .item() (the trainer
        # immediately calls float(v.item()) for W&B serialisation).
        assert hasattr(v, "item"), f"diag[{k!r}] is not a tensor: {type(v)}"
        f = float(v.item())
        assert f == f, f"diag[{k!r}] is NaN"
        assert abs(f) < 1e30, f"diag[{k!r}] = {f} not finite-ish"


# ── Invalid aggregation handling ───────────────────────────────────


def test_invalid_aggregation_raises() -> None:
    """Misconfigured aggregation must fail loud, not silently default."""
    B, L = 2, 8
    pair = _zero_pair(B, L)
    mask = _half_mask(B, L)
    weights = {"rot": 0.0, "pos": 0.0, "seq": 1.0}
    raised = False
    try:
        ipo_loss(pair, mask, weights, 0.05, 100, aggregation="elementwise")
    except ValueError:
        raised = True
    assert raised, "Expected ValueError on bad aggregation, none raised"


# ── Gradient flow sanity ───────────────────────────────────────────


def test_gradient_flows_through_theta_only() -> None:
    """Backward through ipo_loss must populate grads on L_*_theta tensors
    only (and zero/no-grad on L_*_ref). This mirrors the trainer's
    π_θ-trainable / π_ref-frozen contract: in the real trainer the
    π_ref tensors are computed under no_grad, but we replicate the
    invariant here by setting requires_grad only on theta channels.
    """
    B, L = 2, 8
    T = 100
    seq_w_theta = torch.zeros(B, L, requires_grad=True)
    seq_l_theta = torch.zeros(B, L, requires_grad=True)
    zero = lambda: torch.zeros(B, L)  # noqa: E731
    pair = PairLosses(
        w_theta={"rot": zero(), "pos": zero(), "seq": seq_w_theta},
        l_theta={"rot": zero(), "pos": zero(), "seq": seq_l_theta},
        w_ref={"rot": zero(), "pos": zero(), "seq": zero()},
        l_ref={"rot": zero(), "pos": zero(), "seq": zero()},
    )
    mask = _half_mask(B, L)
    weights = {"rot": 0.0, "pos": 0.0, "seq": 1.0}
    loss, _ = ipo_loss(pair, mask, weights, 0.05, T, "residue")
    loss.backward()
    assert seq_w_theta.grad is not None, "seq_w_theta should have grad"
    assert seq_l_theta.grad is not None, "seq_l_theta should have grad"
    # The grads on winner and loser should be equal-and-opposite at iter-0
    # because δ is symmetric in (L_w_θ - L_l_θ) — ∂L/∂L_w_θ = -∂L/∂L_l_θ.
    # This catches a class of "sum vs subtract" bugs in the composite δ.
    assert torch.allclose(
        seq_w_theta.grad + seq_l_theta.grad,
        torch.zeros_like(seq_w_theta.grad),
        atol=1e-5,
    ), (
        f"Expected ∂L/∂L_w_θ = -∂L/∂L_l_θ at iter-0; got "
        f"w_grad={seq_w_theta.grad.mean().item():.4e} "
        f"l_grad={seq_l_theta.grad.mean().item():.4e}"
    )


# ── Trainer dispatch import — the one-line edit in train_dpo.py ──


def test_trainer_dispatch_imports_ipo_loss() -> None:
    """Guards against an accidental rename or removed dispatch.

    The trainer must import ipo_loss; loss_ipo.ipo_loss must accept
    the same parameter signature as abdpo_loss so the dispatch is
    interchangeable.
    """
    import inspect
    from src.dpo.loss import abdpo_loss
    sig_dpo = inspect.signature(abdpo_loss)
    sig_ipo = inspect.signature(ipo_loss)
    # Parameter names must match for the dispatch site to be a true
    # drop-in. (Both have positional + kw params.)
    assert list(sig_dpo.parameters.keys()) == list(sig_ipo.parameters.keys()), (
        f"Signature mismatch:\n  DPO: {list(sig_dpo.parameters.keys())}\n"
        f"  IPO: {list(sig_ipo.parameters.keys())}"
    )
    # Verify train_dpo.py imports ipo_loss (otherwise dispatch is dead).
    train_py = (
        PROJECT_ROOT / "scripts" / "dpo" / "train_dpo.py"
    ).read_text()
    assert "from src.dpo.loss_ipo import ipo_loss" in train_py, (
        "train_dpo.py does not import ipo_loss — dispatch is broken."
    )
    assert "_loss_fn = ipo_loss if objective == \"ipo\" else abdpo_loss" in train_py, (
        "train_dpo.py is missing the objective dispatch one-liner."
    )


# ── Driver ─────────────────────────────────────────────────────────


def _run_all() -> int:
    tests = [
        test_iter0_residue_aggregation_equals_N_tau_squared,
        test_iter0_sequence_aggregation_equals_tau_squared,
        test_zero_loss_when_winner_preferred_to_target_margin,
        test_symmetry_around_tau,
        test_tau_target_diagnostic_matches_formula,
        test_diagnostics_contract,
        test_invalid_aggregation_raises,
        test_gradient_flows_through_theta_only,
        test_trainer_dispatch_imports_ipo_loss,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            failed += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {fn.__name__}: {e.__class__.__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
