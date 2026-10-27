"""Explicit scientific stages; immutable task IDs never depend on worker count or paths."""

from dataclasses import asdict
from typing import cast

from common.experiments.registry import TaskRegistry
from common.io.json import fingerprint
from lifted.types import Task
from theory.parameters import Parameters

from .protocol import OBSERVATION_TAUS, PROTOCOL, RATIO_GRID, RATIOS, sources

REGISTRY = TaskRegistry({"smoke": "smoke", "pilot": "pilot", "formal": "formal"})


class Design:
    def __init__(self, profile):
        self.profile, self.tasks, self.stages = profile, {}, {}

    def add(
        self,
        stage,
        kind,
        p,
        *,
        parent=None,
        at=0,
        initial=0,
        mother=0,
        seed=20261027,
        paths=1,
        start=0,
        steps=0,
        observe=(),
        variant="stochastic",
        layer="naive",
        cohort=None,
        initial_law="uniform",
        bridge_agents=0,
        group_center=0.35,
    ):
        task = dict(
            kind=kind,
            parameters=asdict(p),
            cohort=cohort or self.profile,
            initial=initial,
            mother=mother,
            seed=seed,
            replicates=list(range(start, start + paths)),
            horizon_steps=steps,
            observe_steps=list(observe),
            source_task=parent,
            source_step=at,
            variant=variant,
            layer=layer,
            initial_law=initial_law,
            bridge_agents=bridge_agents,
            group_center=group_center,
        )
        task["id"] = kind + "_" + fingerprint(task)[:24]
        identifier = task["id"]
        if identifier not in self.tasks:
            self.tasks[identifier] = cast(Task, task)
            self.stages.setdefault(stage, []).append(identifier)
        return identifier

    def freeze(self, budget=None):
        return dict(
            schema=1,
            profile=self.profile,
            tasks=list(self.tasks.values()),
            stages=[dict(name=name, task_ids=ids) for name, ids in self.stages.items()],
            protocol=PROTOCOL,
            sources=sources(),
            budget=budget or {},
        )


def parameters(ratio=2, *, n=500, bins=161, epsilon=0.4):
    return Parameters(
        dynamics="hk",
        recommender="random",
        epsilon=epsilon,
        alpha=0.02,
        ratio=ratio,
        degree=15,
        recommendations=10,
        bins=bins,
        quadrature=33,
        population=n,
        dt=1,
        horizon=120,
        record_tau=0.1,
    )


def chunked(design, stage, kind, p, *, paths, block=8, start=0, **options):
    return [
        design.add(stage, kind, p, paths=min(block, paths - offset), start=start + offset, **options)
        for offset in range(0, paths, block)
    ]


def validation_design(profile):
    REGISTRY.select(profile)
    if profile == "formal":
        raise ValueError("formal plans require authenticated pilot timings")
    d = Design(profile)
    smoke = profile == "smoke"
    n, bins = (40, 15) if smoke else (500, 161)
    ratios = (2.0,) if smoke else RATIOS
    for index, ratio in enumerate(ratios):
        p = parameters(ratio, n=n, bins=bins)
        root = d.add("initial", "initial", p, initial=index)
        atlas_steps = 8 if smoke else 1500
        at = 4 if smoke else 200
        atlas = d.add(
            "micro_atlas", "micro_atlas", p, parent=root, steps=atlas_steps, observe=(0, at, atlas_steps), initial=index
        )
        horizon = 24 if smoke else 24000
        d.add("micro_truth", "micro_tail", p, parent=atlas, at=atlas_steps, steps=horizon, initial=index)
        chunked(
            d,
            "micro_probe",
            "micro_probe",
            p,
            parent=atlas,
            at=at,
            paths=4 if smoke else 4,
            steps=horizon,
            initial=index,
        )
        d.add(
            "micro_moments", "micro_moments", p, parent=atlas, at=at, paths=4 if smoke else 32, steps=1, initial=index
        )
        grids = (15,) if smoke else (81, 161)
        if not smoke:
            for query_at in (0, atlas_steps):
                chunked(
                    d, "micro_probe", "micro_probe", p, parent=atlas, at=query_at, paths=4, steps=horizon, initial=index
                )
                for b in grids:
                    mesh = parameters(ratio, n=n, bins=b)
                    chunked(
                        d,
                        "meso_probe",
                        "meso_probe",
                        mesh,
                        parent=atlas,
                        at=query_at,
                        paths=2,
                        block=2,
                        steps=horizon,
                        initial=index,
                    )
        for b in grids:
            mesh = parameters(ratio, n=n, bins=b)
            for variant in (
                ("stochastic", "deterministic", "rewire_only", "transport_only")
                if smoke or ratio == 2
                else ("stochastic",)
            ):
                # Long stochastic paths measure absorption cost; short controls measure stepping cost.
                steps = horizon if variant == "stochastic" else (8 if smoke else 1500)
                chunked(
                    d,
                    "meso_probe",
                    "meso_probe",
                    mesh,
                    parent=atlas,
                    at=at,
                    paths=2 if smoke or variant == "stochastic" else 1,
                    block=2,
                    steps=steps,
                    initial=index,
                    variant=variant,
                )
            d.add(
                "meso_moments",
                "meso_moments",
                mesh,
                parent=atlas,
                at=at,
                paths=4 if smoke else 32,
                steps=1,
                initial=index,
            )
        meso = d.add(
            "meso_atlas",
            "meso_atlas",
            p,
            parent=atlas,
            at=at,
            steps=8 if smoke else 200,
            observe=(0, 4 if smoke else 100),
            initial=index,
        )
        d.add(
            "meso_conditional",
            "meso_probe",
            p,
            parent=meso,
            at=at + (4 if smoke else 100),
            paths=2,
            steps=8 if smoke else 200,
            initial=index,
        )
        d.add(
            "measure",
            "measure",
            p,
            parent=atlas,
            at=at,
            steps=8 if smoke else 200,
            observe=(0, 1, 8 if smoke else 200),
            initial=index,
            variant="deterministic",
        )
        # Base/naive paired equality is checked on a complete shared projected state.
        d.add(
            "equivalence",
            "meso_moments",
            p,
            parent=atlas,
            at=at,
            paths=4 if smoke else 32,
            steps=1,
            initial=index,
            layer="base",
        )
    if not smoke:
        for n in (250, 1000, 2000):
            p = parameters(n=n)
            root = d.add("initial", "initial", p, initial=20 + n)
            d.add("micro_size", "micro_probe", p, parent=root, paths=2, steps=400, initial=20 + n)
            d.add("meso_size", "meso_probe", p, parent=root, paths=1, steps=200, initial=20 + n)
        p = parameters(bins=241)
        root = d.add("initial", "initial", p, initial=99)
        d.add("meso_fine", "meso_probe", p, parent=root, paths=1, steps=200, initial=99)
    return d.freeze()


def formal_design(*, initials, mothers, micro_paths, meso_paths, budget):
    d = Design("formal")
    for ratio in RATIOS:
        p = parameters(ratio)
        for initial in range(initials):
            for mother in range(mothers):
                root = d.add("initial", "initial", p, initial=initial, mother=mother, cohort="population")
                atlas = d.add(
                    "micro_atlas",
                    "micro_atlas",
                    p,
                    parent=root,
                    initial=initial,
                    mother=mother,
                    steps=6000,
                    observe=[round(t / 0.02) for t in OBSERVATION_TAUS],
                    cohort="population",
                )
                d.add(
                    "micro_truth",
                    "micro_tail",
                    p,
                    parent=atlas,
                    at=6000,
                    steps=24000,
                    initial=initial,
                    mother=mother,
                    cohort="population",
                )
                for tau in OBSERVATION_TAUS:
                    if tau == 0 and mother > 0:
                        continue  # Identical physical initial state; share its probability estimate.
                    at = round(tau / 0.02)
                    chunked(
                        d,
                        "micro_probe",
                        "micro_probe",
                        p,
                        parent=atlas,
                        at=at,
                        initial=initial,
                        mother=mother,
                        paths=micro_paths,
                        steps=24000,
                        cohort="population",
                    )
                    if initial < 2 and mother == 0 and tau in (4, 16, 30):
                        d.add(
                            "micro_moments",
                            "micro_moments",
                            p,
                            parent=atlas,
                            at=at,
                            initial=initial,
                            paths=256,
                            steps=1,
                            cohort="population",
                        )
                        d.add(
                            "measure",
                            "measure",
                            p,
                            parent=atlas,
                            at=at,
                            steps=6000,
                            observe=(0, 1, 200, 800, 1500, 6000),
                            initial=initial,
                            cohort="population",
                            variant="deterministic",
                        )
                        for bins in (81, 161, 241):
                            mesh = parameters(ratio, bins=bins)
                            paths = meso_paths if bins == 161 else max(16, meso_paths // 4)
                            chunked(
                                d,
                                "meso_probe",
                                "meso_probe",
                                mesh,
                                parent=atlas,
                                at=at,
                                initial=initial,
                                paths=paths,
                                steps=24000,
                                cohort="population",
                            )
                            d.add(
                                "meso_moments",
                                "meso_moments",
                                mesh,
                                parent=atlas,
                                at=at,
                                initial=initial,
                                paths=128,
                                steps=1,
                                cohort="population",
                            )
                        for variant in ("deterministic", "rewire_only", "transport_only"):
                            if initial != 0 or tau not in (4, 16):
                                continue
                            chunked(
                                d,
                                "noise_controls",
                                "meso_probe",
                                p,
                                parent=atlas,
                                at=at,
                                initial=initial,
                                paths=1 if variant == "deterministic" else 8,
                                steps=6000,
                                variant=variant,
                                cohort="mechanism",
                            )
                if initial == 0 and mother == 0:
                    meso = d.add(
                        "meso_atlas",
                        "meso_atlas",
                        p,
                        parent=root,
                        steps=6000,
                        observe=(0, 200, 800, 1500, 6000),
                        initial=initial,
                        cohort="mechanism",
                    )
                    for at in (200, 800, 1500):
                        chunked(
                            d,
                            "meso_conditional",
                            "meso_probe",
                            p,
                            parent=meso,
                            at=at,
                            paths=meso_paths,
                            steps=24000,
                            initial=initial,
                            cohort="mechanism",
                        )
    # Prespecified mechanism/size extensions stay separate from population timing.
    for ratio in RATIO_GRID:
        p = parameters(ratio, bins=81)
        root = d.add("initial", "initial", p, initial=500, cohort="mechanism")
        chunked(
            d,
            "meso_broad",
            "meso_probe",
            p,
            parent=root,
            paths=max(16, meso_paths // 8),
            steps=24000,
            initial=500,
            cohort="mechanism",
        )
    for n in (250, 1000, 2000):
        p = parameters(n=n)
        root = d.add("initial", "initial", p, initial=600 + n, cohort="mechanism")
        atlas = d.add(
            "micro_atlas",
            "micro_atlas",
            p,
            parent=root,
            initial=600 + n,
            steps=1500,
            observe=(200, 800, 1500),
            cohort="mechanism",
        )
        for at in (200, 800):
            if n <= 1000:
                chunked(
                    d,
                    "micro_size",
                    "micro_probe",
                    p,
                    parent=atlas,
                    at=at,
                    initial=600 + n,
                    paths=32,
                    steps=24000,
                    cohort="mechanism",
                )
            chunked(
                d,
                "meso_size",
                "meso_probe",
                p,
                parent=atlas,
                at=at,
                initial=600 + n,
                paths=max(16, meso_paths // 4),
                steps=24000,
                cohort="mechanism",
            )
    for bridge in (1, 4, 16):
        p = parameters()
        root = d.add(
            "initial", "initial", p, initial=400, cohort="mechanism", initial_law="bridge", bridge_agents=bridge
        )
        atlas = d.add(
            "micro_atlas",
            "micro_atlas",
            p,
            parent=root,
            initial=400,
            steps=1500,
            observe=(0, 200, 800, 1500),
            cohort="mechanism",
        )
        for at in (0, 200, 800):
            chunked(
                d,
                "micro_bridge",
                "micro_probe",
                p,
                parent=atlas,
                at=at,
                initial=400,
                paths=32,
                steps=24000,
                cohort="mechanism",
            )
            chunked(
                d,
                "meso_bridge",
                "meso_probe",
                p,
                parent=atlas,
                at=at,
                initial=400,
                paths=max(16, meso_paths // 4),
                steps=24000,
                cohort="mechanism",
            )
    return d.freeze(budget)
