"""Strict JSONL transport contract and explicit microscopic execution policies."""

from copy import deepcopy

from common.io.json import canonical

# Historical audits share initial RNG draws and enforce nominal-degree rows.
# Apply these defaults at transport time so frozen task identities remain intact.
AUDIT_POLICIES = {
    "initial_rng": "shared",
    "projection": "fixed_degree_rows",
    "record_policy": "audit",
    "energy_normalization": "nominal",
}
# Population comparisons separate opinion/graph streams and retain actual graph
# mass and edge normalization; compact frames avoid saving unused edge fields.
POPULATION_POLICIES = {
    "initial_rng": "split",
    "projection": "actual",
    "record_policy": "population",
    "energy_normalization": "actual",
}


def prepare_task(payload):
    """Deep-copy a payload and supply audit policies for missing microscopic options; preserve explicit overrides and frozen task bytes."""
    value = deepcopy(payload)
    if value.get("kind") == "micro":
        value = {**AUDIT_POLICIES, **value}
    return value


def task_line(payload):
    """Return one canonical JSONL request after applying transport-only policy defaults."""
    return canonical(prepare_task(payload)) + "\n"


def read_response(text, identifier):
    """Decode one JSON response and return result; require matching id and status=ok, otherwise raise RuntimeError."""
    import json

    response = json.loads(text)
    if response.get("id") != identifier or response.get("status") != "ok":
        raise RuntimeError(f"{identifier}: {response.get('error', 'invalid runtime response')}")
    return response["result"]


FAMILIES = ("micro", "clocks", "operator", "kernel")
