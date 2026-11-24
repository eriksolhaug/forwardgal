"""Ray-polygon geometry for exact pixel integration (JAX).

All positions are 2-D, in any consistent length unit, in a frame whose origin
is the centre of the rays (e.g. the centre of a Sersic profile).
"""

from __future__ import annotations

import jax.numpy as jnp


def polygon_signed_area(vertices):
    """Signed area of a polygon (shoelace formula).

    Parameters
    ----------
    vertices : (M, 2) array
        Polygon vertices in order (either winding direction).

    Returns
    -------
    float
        Positive for counter-clockwise order, negative for clockwise.
    """
    x, y = vertices[:, 0], vertices[:, 1]
    return 0.5 * jnp.sum(x * jnp.roll(y, -1) - jnp.roll(x, -1) * y)


def contains_origin(vertices):
    """True if the origin lies strictly inside a convex polygon."""
    a = vertices
    edge = jnp.roll(vertices, -1, axis=0) - a
    # z-component of edge x (origin - a): same sign for all edges iff inside.
    cross = edge[:, 0] * (-a[:, 1]) - edge[:, 1] * (-a[:, 0])
    return jnp.all(cross > 0) | jnp.all(cross < 0)


def angular_segments(vertices):
    """Split the angles a convex polygon covers (seen from the origin) at its corners.

    Between two consecutive corner directions, every ray enters and leaves the
    polygon through the same two edges, so t_in(phi) and t_out(phi) are smooth
    there; at a corner direction they have a kink. Integrating segment by
    segment with Gauss-Legendre quadrature therefore converges very fast.

    Parameters
    ----------
    vertices : (M, 2) array
        Convex polygon, any winding order. Must not have a vertex exactly at
        the origin.

    Returns
    -------
    lo, hi : (M,) arrays
        Segment limits in radians. If the origin is inside, the M segments
        cover the full circle. Otherwise the polygon spans less than pi and
        is covered by M - 1 segments; the last segment has zero length.
    """
    m = vertices.shape[0]
    inside = contains_origin(vertices)
    centroid = jnp.mean(vertices, axis=0)
    # Avoid atan2(0, 0) (undefined gradient) in the branch that is not used.
    centroid = jnp.where(inside, jnp.array([1.0, 0.0]), centroid)
    phi_c = jnp.arctan2(centroid[1], centroid[0])
    # Corner angles relative to the centroid direction, wrapped to [-pi, pi):
    # for an outside polygon this keeps them from wrapping around +-pi.
    rel = jnp.arctan2(vertices[:, 1], vertices[:, 0]) - phi_c
    rel = jnp.sort((rel + jnp.pi) % (2.0 * jnp.pi) - jnp.pi)
    closed = jnp.concatenate([rel, rel[:1] + 2.0 * jnp.pi])  # cyclic, inside
    open_ = jnp.concatenate([rel, rel[-1:]])  # last segment empty, outside
    edges = phi_c + jnp.where(inside, closed, open_)
    return edges[:m], edges[1:]


def ray_polygon_range(vertices, phi):
    """Where rays from the origin enter and leave a convex polygon.

    Each ray is p(t) = t * (cos phi, sin phi), t >= 0. The polygon is the
    intersection of the half-planes n_i . p <= c_i (n_i = outward normal of
    edge i, c_i = n_i . A_i with A_i a vertex on that edge). Along the ray this
    reads t (n_i . d) <= c_i, which bounds t from above (n_i . d > 0), from
    below (n_i . d < 0), or is always/never true (n_i . d = 0).

    Parameters
    ----------
    vertices : (M, 2) array
        Vertices of a CONVEX polygon, in order (either winding direction).
    phi : (N,) array
        Ray angles in radians, measured from the +x axis of this frame.

    Returns
    -------
    t_in, t_out : (N,) arrays
        Distances along each ray where it enters and leaves the polygon.
        t_in = 0 if the origin is inside. Both are 0 if the ray misses.
    """
    a = vertices
    b = jnp.roll(vertices, -1, axis=0)
    edge = b - a  # (M, 2)
    # Outward normal for counter-clockwise order is (ey, -ex); flip if clockwise.
    orient = jnp.sign(polygon_signed_area(vertices))
    normal = orient * jnp.stack([edge[:, 1], -edge[:, 0]], axis=1)  # (M, 2)
    c = jnp.sum(normal * a, axis=1)  # (M,)

    d = jnp.stack([jnp.cos(phi), jnp.sin(phi)], axis=1)  # (N, 2)
    nd = d @ normal.T  # (N, M): n_i . d for every ray and edge

    # Guard the division: where nd ~ 0 use a dummy denominator, and handle that
    # case separately. (Dividing by 0 inside jnp.where would still produce NaN
    # gradients, even in the branch that is not selected.)
    parallel = jnp.abs(nd) < 1e-14
    safe_nd = jnp.where(parallel, 1.0, nd)
    bound = c / safe_nd  # (N, M)

    # Parallel edges give no bound on t (only the all-or-nothing test below).
    upper = jnp.where((nd > 0) & ~parallel, bound, jnp.inf)
    lower = jnp.where((nd < 0) & ~parallel, bound, -jnp.inf)
    # A parallel edge with c < 0 excludes the whole ray.
    excluded = jnp.any(parallel & (c < 0), axis=1)

    t_in = jnp.maximum(0.0, jnp.max(lower, axis=1))
    t_out = jnp.min(upper, axis=1)

    hit = (t_out > t_in) & ~excluded
    return jnp.where(hit, t_in, 0.0), jnp.where(hit, t_out, 0.0)
