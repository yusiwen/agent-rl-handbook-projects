"""The shared contract for the six projects.

Standard library only, on purpose.  See ``AGENTS.md`` non-negotiable 2: a shared
definition of a rollout that imports a training framework is not shareable.

Modules:

``schema``    the rollout row every project writes
``mask``      loss-mask construction, plus the fault check that proves it is right
``verifier``  the ``Verifier`` protocol, the registry, the coverage report
``fusion``    merging rule and model reward terms, and spotting their conflicts
``runs``      run headers, per-round records, stable sample ids
``checks``    the five assertions the book makes to the reader
"""

from .schema import ROLES, Rollout, Segment
from .verifier import MissingVerifier, Outcome, Verifier, VerifierRegistry

__all__ = [
    "ROLES",
    "Rollout",
    "Segment",
    "MissingVerifier",
    "Outcome",
    "Verifier",
    "VerifierRegistry",
]
