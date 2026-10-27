"""Study-owned family/ratio scheduling priorities; task identities are unchanged."""


def family(task):
    if task["kind"] == "initial":
        return "prepare"
    if task["kind"] in ("micro_atlas", "micro_probe", "micro_tail", "micro_extend"):
        return "micro"
    if task["kind"] in ("meso_atlas", "meso_probe", "meso_extend") and task["variant"] == "stochastic":
        return "meso"
    return "diagnostics"


def fair_key(task):
    """Each family/ratio gets a turn; first chunks cover states before extra replicas."""
    lane = (family(task), task["parameters"]["ratio"])
    priority = (
        min(task["replicates"]),
        task["cohort"] == "mechanism",
        task["initial"],
        task["mother"],
        task["source_step"],
        task["parameters"]["bins"] != 161,
        task["kind"],
    )
    return lane, priority
