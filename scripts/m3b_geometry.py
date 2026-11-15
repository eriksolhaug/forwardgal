"""Milestone 3b, step 1: rays from a profile centre clipped to pixels.

Draws the part of each ray that lies inside two pixels: one containing the
centre (rays start at the centre) and one beside it (rays enter and leave).
Also checks area = 1/2 * integral (t_out^2 - t_in^2) dphi against the
shoelace formula.

Run from the repository root:
    python scripts/m3b_geometry.py
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import forwardgal  # noqa: F401  (enables 64-bit JAX)
from forwardgal.geometry import polygon_signed_area, ray_polygon_range

centre = np.array([0.13, -0.27])  # profile centre inside pixel (0, 0), pixel units
pixels = {
    "contains centre": np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]),
    "neighbour": np.array([[0.5, -0.5], [1.5, -0.5], [1.5, 0.5], [0.5, 0.5]]),
}

phi_plot = np.linspace(0, 2 * np.pi, 48, endpoint=False)
nphi = 4000
phi = (np.arange(nphi) + 0.5) / nphi * 2 * np.pi

fig, ax = plt.subplots(figsize=(7, 4.5))
for (label, square), colour in zip(pixels.items(), ["C0", "C1"]):
    verts = jnp.array(square - centre)  # frame centred on the profile centre
    t_in, t_out = ray_polygon_range(verts, jnp.array(phi_plot))
    for p, a, b in zip(phi_plot, np.asarray(t_in), np.asarray(t_out)):
        if b > 0:
            d = np.array([np.cos(p), np.sin(p)])
            seg = centre + np.outer([a, b], d)
            ax.plot(seg[:, 0], seg[:, 1], color=colour, lw=0.8)
    closed = np.vstack([square, square[:1]])
    ax.plot(closed[:, 0], closed[:, 1], "k-", lw=1.5)

    ti, to = ray_polygon_range(verts, jnp.array(phi))
    polar_area = 0.5 * float(jnp.sum(to**2 - ti**2)) * 2 * np.pi / nphi
    print(f"{label:16s}: polar area = {polar_area:.6f}, shoelace = {abs(float(polygon_signed_area(verts))):.6f}")

ax.plot(*centre, "r*", ms=12, label="profile centre")
ax.set_aspect("equal")
ax.set_xlabel("x [pixels]")
ax.set_ylabel("y [pixels]")
ax.legend(loc="upper left")
ax.set_title("ray segments inside each pixel")
fig.savefig("outputs/m3b_geometry.png", dpi=110, bbox_inches="tight")
print("saved figure    : outputs/m3b_geometry.png")
plt.show()
