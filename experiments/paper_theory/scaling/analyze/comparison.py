"""Compatible small-alpha and nominal-population comparisons in scaled time."""

from collections import defaultdict

from theory.scaling import curve_distance


def scaling_rows(cases, *, population=False):
    """Return channel-wise curve gaps within matching physical/numerical groups; reference the smallest alpha or, for population=True, the largest N."""
    groups = defaultdict(list)
    for case in cases:
        # Compare only matching laws, kernels, confidence geometry and resolutions.
        group = (
            case["dynamics"],
            case.get("recommender", "random"),
            case.get("steepness", 1),
            case.get("quadrature", 9),
            case["epsilon"],
            round(case["r"], 9),
            case.get("bins", 161),
            case.get("degree", 15),
            case.get("recommendations", 10),
            case.get("dt", 1),
            case.get("noise_diffusion", 0),
            case.get("confidence_mode", "cell_average"),
            case.get("initial_law", "uniform[-1,1]"),
        )
        if population:
            group += (case["alpha"],)
        else:
            group += (case.get("population", 500),)
        groups[group].append(case)
    rows = []
    for members in groups.values():
        if len(members) < 2:
            continue
        reference = (
            max(members, key=lambda c: c["population"]) if population else min(members, key=lambda c: c["alpha"])
        )
        for case in members:
            if case is reference:
                continue
            for name in ("node_energy", "edge_energy"):
                rows.append(
                    {
                        "dynamics": case["dynamics"],
                        "recommender": case.get("recommender", "random"),
                        "steepness": case.get("steepness", 1),
                        "quadrature": case.get("quadrature", 9),
                        "epsilon": case["epsilon"],
                        "r": case["r"],
                        "alpha": case["alpha"],
                        "reference_alpha": reference["alpha"],
                        "population": case.get("population", 500),
                        "reference_population": reference.get("population", 500),
                        "energy": name,
                        **curve_distance(reference["tau"], reference[name], case["tau"], case[name]),
                    }
                )
    return rows
