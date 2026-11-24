"""Milestone 12d: recover Fourier modes (m = 1, 3, 4) from synthetic lensed data.

Setup of tests/test_fit_modes.py: truth with modes, start with all modes at
0, one LM fit, Laplace errors. Prints truth vs fit for a_m and phi_m.
Needs fourier_flux_factor (M12b).

Run from the repository root:
    python scripts/m12d_fit_modes.py
"""

import sys

sys.path.insert(0, "tests")

import forwardgal  # noqa: E402,F401
from forwardgal.fit import laplace_errors  # noqa: E402
from test_fit_modes import FLUX, MODES, SHAPE, fitted  # noqa: E402

model, res, _ = fitted.__wrapped__()
lap = laplace_errors(model, res)
err = {d[0].replace(" (derived)", ""): d[2] for d in lap.derived}
err.update(dict(zip(lap.names, lap.sigma)))
v = res.values["host"]
print(res.describe(model).splitlines()[0])
print(f"{'parameter':10s} {'truth':>8s} {'fit':>9s} {'sigma':>8s}")
for m, (a, phi) in MODES.items():
    print(f"{'a' + str(m):10s} {a:8.3f} {float(v[f'a{m}']):9.4f} {err[f'host.a{m}']:8.4f}")
    print(f"{'phi' + str(m):10s} {phi:8.1f} {float(v[f'phi{m}']):9.2f} {err[f'host.phi{m}']:8.2f}")
print(f"{'R_e':10s} {SHAPE['R_e']:8.3f} {float(v['R_e']):9.4f} {err['host.R_e']:8.4f}")
print(f"{'flux':10s} {FLUX:8.1f} {res.amplitudes['host']:9.2f} {err['amp:host']:8.2f}")
