"""Historical shard readers and tail-normalized survival/quantile interpolation."""

import json

import numpy as np

from experiments.paper_theory.release_clocks.analyze import config
from theory.release.curves import norm


def cases(alpha_only=False):
    """Yield normalized node/edge curves from the ten archived raw shards; alpha_only restricts the alpha=.01 pilot."""
    paths = sorted((config.SOURCE / "raw_shards").glob("*.npz"))
    if len(paths) != 10:
        raise FileNotFoundError(
            f"Expected 10 original raw shards in {config.SOURCE / 'raw_shards'}; see the artifact README for regeneration instructions."
        )
    for path in paths:
        if alpha_only and (not path.name.endswith("_0.01.npz")):
            continue
        z = np.load(path)
        dyn = path.name.split("_")[0]
        alpha = (
            float(json.loads(str(z["protocol"]))["alpha"])
            if "alpha" in json.loads(str(z["protocol"]))
            else float(path.stem.split("_alpha_")[1])
        )
        for i, eps in enumerate(z["epsilon"]):
            for j, r in enumerate(z["q_over_alpha"]):
                n = int(z["lengths"][i, j])
                t = z["tau"][i, j, :n].copy()
                node = z["node_energy"][i, j, :n].copy()
                edge = z["edge_energy"][i, j, :n].copy()
                if t[1] > alpha + 1e-12:
                    t = np.insert(t, 1, alpha)
                    node = np.insert(node, 1, node[0] + z["first_step_delta"][i, j, 0])
                    edge = np.insert(edge, 1, edge[0] + z["first_step_delta"][i, j, 1])
                sn, cn, an = norm(node)
                se, ce, ae = norm(edge)
                yield dict(
                    dynamics=dyn,
                    alpha=alpha,
                    epsilon=float(eps),
                    r=float(r),
                    converged=bool(z["converged"][i, j]),
                    t=t,
                    sn=sn,
                    se=se,
                    c=ce,
                    a=ae,
                    cn=cn,
                    an=an,
                )


def key(c):
    """Return the rounded physical case key used to join archived fit tables."""
    return (c["dynamics"], round(c["alpha"], 12), c["epsilon"], round(c["r"], 9))
