"""Completed historical audits retained by Supplemental revision_validation.

These reproduce the 762-job revision and 346-job follow-up batches only.
Current generation uses evidence.design; there are no historical smoke,
minimum, extended, robustness or alternative sweep profiles.
"""

from .kernel_validation import design as kernel
from .micro_validation import design as micro
from .operator_validation import design as operator
from .release_clocks import design as clocks

FAMILIES = ("micro", "clocks", "operator", "kernel")


def generate(*, preset="validation"):
    if preset == "validation":
        jobs = [
            *micro.validation_cases(),
            *clocks.validation_cases(),
            *operator.validation_cases(),
            *kernel.validation_cases(),
        ]
    elif preset == "ensembles":
        jobs = [*micro.ensemble_cases(), *clocks.ensemble_cases()]
    else:
        raise ValueError("only the completed validation/ensembles historical batches are retained")
    if len({j["id"] for j in jobs}) != len(jobs):
        raise ValueError("duplicate historical job identities")
    return jobs
