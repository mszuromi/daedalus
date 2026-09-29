# Pre-M1 Phase J fixtures (legacy copies, never refrozen)

These four files are byte-for-byte copies of the frozen Phase J regression
fixtures in the parent directory. They were copied at milestone M0.3 of the
integration speed-up plan, before any number-moving change (M1 onward) landed.

| file | sha256 |
|---|---|
| `spike_reset_k1_ell1.npz` | `7814492635b59703f4209771e8b0a555817dcf18f8ab65273b9fc2d809eaf079` |
| `spike_reset_k2_ell0.npz` | `50e50379c3f8e73778b01ab5f278acbb4839b905bbe2eca62d40b75bfa6f205d` |
| `spike_reset_k2_ell1.npz` | `2fafe8fad0107bceb25cbf40872667fc3e8757c56d834713af7fca3c106d6ec7` |
| `quad_exp_k2_ell0.npz` | `ac2d0c0ec008ac4428b6f5f539af79e1ede530751080d9d6f9b1295d0844993e` |

Source: `tests/phase_j_refactor_fixtures/*.npz` at commit `b2bf559`. The
copies are identical to the blobs in that commit.

These are the **pre-M1 values**. They include the Θ(0)=1 behaviour of the m=2
polygon path. For example, `spike_reset_k2_ell1` still carries that bug.

**Never refreeze these files.**
- The parent-directory fixtures may be refrozen once, at M9. These copies stay
  as they are.
- From M1, the legacy-umbrella test (`DAEDALUS_PHASE_J_LEGACY=1`) will compare
  against them. Until then, `tests/test_phase_j_legacy_baseline.py` only
  checks that they are byte-identical to the hashes above.
- The per-milestone fixture delta report compares against them
  (`tests/tools/phase_j_subset_diff.py --fixture-report`).

The zoo-wide pre-change baseline is kept separately, in
`tests/fixtures/phase_j_legacy_baseline.npz`.
