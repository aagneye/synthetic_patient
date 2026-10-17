"""Method C: the diffusion plugin contract (stub, not implemented).

SYNTHETIC. In silico cohort validation. Not clinical evidence.

This module deliberately generates nothing. It defines the interface a
diffusion generator must satisfy to be plugged in later, and it refuses to run
until one is registered.

Why it is a stub
----------------

Kenneweg et al. 2025 (PMID 40662828) compress a genotype by splitting the
genome into genes, running PCA inside each gene down to 1-8 components, and
training a U-Net diffusion model on the compressed matrix. Their results are
strong on their own tasks: about 89 percent average recovery rate, and with
only 5 percent of real data plus synthetic data, 1000 Genomes population
classification accuracy rises from 29.01 to 83.98.

None of that makes it safe for the DPYD region here, for four reasons that the
build brief sets out and this stub enforces:

1. **PCA compression favours common variation.** The variants that decide a
   DPYD call are rare: ``*2A`` and ``*13`` sit under 1 percent. One to eight
   components per gene can wash them out. A cohort that cannot reproduce
   ``*2A`` at the right frequency cannot test a DPYD engine.
2. **The paper's downstream tasks are not this task.** Case/control status and
   which-of-26-populations say nothing about whether pharmacogene haplotype
   structure survived.
3. **Training data does not exist yet.** The method needs thousands of
   individual-level genotypes and a GPU. Public GenomeIndia access is summary
   statistics, not individual genomes.
4. **Memorisation is a live risk.** The authors say they cannot fully guarantee
   the original data cannot be inferred. Training on Indian individual-level
   genomes without a privacy review would be an ethics and a legal problem.

So the deterministic engine never depends on a generative model. When a
generator does arrive, it is for the **background** genome, spliced around a
DPYD region that still comes from coalescent simulation plus injection, and it
has to clear the Part 8 gates against the coalescent arm before it is used for
anything.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


class DiffusionNotAvailable(NotImplementedError):
    """Raised when a diffusion backend is requested but none is registered."""


@runtime_checkable
class BackgroundGenerator(Protocol):
    """What a plugged-in diffusion generator must provide.

    Implementations generate **background** haplotypes only. The DPYD region is
    supplied separately by :mod:`synthpgx.genotypes` and spliced in, which is
    what preserves the rare clinically relevant variants.
    """

    #: Identifier written into ``generation_method`` provenance.
    name: str

    def sample_background(
        self,
        n_individuals: int,
        population: str,
        seed: int,
    ) -> np.ndarray:
        """Return an ``(n_individuals, n_sites)`` dosage matrix in 0/1/2."""
        ...

    def site_positions(self) -> np.ndarray:
        """GRCh38 positions of the columns ``sample_background`` returns."""
        ...

    def privacy_report(self) -> dict[str, Any]:
        """Metrics the Part 8.3 privacy gate needs.

        Must include ``nnaa``, ``min_nearest_neighbour_distance`` and
        ``membership_inference_accuracy``, each measured against the cohort the
        generator was trained on. A generator that cannot report these cannot
        be used on controlled-access data.
        """
        ...


_REGISTRY: dict[str, BackgroundGenerator] = {}


def register(generator: BackgroundGenerator) -> None:
    """Register a diffusion backend under its ``name``."""
    if not isinstance(generator, BackgroundGenerator):
        raise TypeError(
            "generator does not satisfy the BackgroundGenerator protocol: it "
            "must provide name, sample_background, site_positions and "
            "privacy_report"
        )
    _REGISTRY[generator.name] = generator


def available() -> list[str]:
    return sorted(_REGISTRY)


def get(name: str | None = None) -> BackgroundGenerator:
    if not _REGISTRY:
        raise DiffusionNotAvailable(
            "Method C (diffusion) is a stub in this build. No background "
            "generator is registered. Register one with "
            "synthpgx.diffusion.register(...) and re-run the Part 8 gates "
            "against the coalescent arm before using it. See the module "
            "docstring for why the DPYD region must not come from a diffusion "
            "model."
        )
    if name is None:
        return _REGISTRY[sorted(_REGISTRY)[0]]
    if name not in _REGISTRY:
        raise DiffusionNotAvailable(
            f"no diffusion backend named {name!r}; registered: {available()}"
        )
    return _REGISTRY[name]


def splice_background(
    pgx_dosages: np.ndarray,
    pgx_positions: np.ndarray,
    background_dosages: np.ndarray,
    background_positions: np.ndarray,
    protected_region: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Combine a generated background with a coalescent-derived PGx region.

    Background columns falling inside ``protected_region`` are dropped, so the
    DPYD interval is carried entirely by ``pgx_dosages``. This is the hybrid the
    build brief recommends: genome-wide realism from the generator, clinically
    critical rare variants from coalescent simulation plus injection.
    """
    if pgx_dosages.shape[0] != background_dosages.shape[0]:
        raise ValueError(
            "pgx and background matrices must cover the same individuals: "
            f"{pgx_dosages.shape[0]} vs {background_dosages.shape[0]}"
        )
    low, high = protected_region
    keep = (background_positions < low) | (background_positions > high)
    positions = np.concatenate([background_positions[keep], pgx_positions])
    dosages = np.concatenate([background_dosages[:, keep], pgx_dosages], axis=1)
    order = np.argsort(positions)
    return dosages[:, order], positions[order]
