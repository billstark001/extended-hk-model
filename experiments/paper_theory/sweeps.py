"""Compose study-owned mesoscopic sweeps and preserve their task identities."""

from dataclasses import asdict


def task(suite, p):
    """Return one measure-sweep task with a parameter-derived readable ID and serialized physical/numerical settings."""
    p.validate()
    identifier = (
        f"{suite}_{p.dynamics}_{p.recommender}_e{p.epsilon:g}_r{p.ratio:g}_a{p.alpha:g}_n{p.population}_b{p.bins}"
    )
    return {"id": identifier, "suite": suite, "parameters": asdict(p)}


def designs(suite="scaling", profile="paper"):
    """Dispatch scaling, release or population_closure designs for the requested paper/smoke profile."""
    from .micro_validation.design import closure_designs
    from .release_clocks.design import sweep_designs
    from .scaling.design import designs as scaling

    factory = {"scaling": scaling, "release": sweep_designs, "population_closure": closure_designs}.get(suite)
    if factory is None:
        raise ValueError("unknown sweep suite")
    return factory(profile)
