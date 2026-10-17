"""Method A: coalescent simulation of the DPYD region plus variant injection.

SYNTHETIC. In silico cohort validation. Not clinical evidence.

Two stages, as the build brief requires (6.1 and 6.4):

1. **Background.** One ``msprime`` coalescent over the DPYD region only, under a
   shared demography so populations have realistic Fst, recombination and
   linkage disequilibrium. Simulating the gene region rather than a genome is
   what keeps this on a laptop.

2. **Injection.** Clinically relevant variants are placed afterwards at their
   sourced per-population frequencies, because a neutral coalescent will not
   reproduce ascertained variants at the right frequency by itself.

The injection is clade-based: for a target allele count the simulator finds a
node in the local tree whose descendant count within that population is closest
to the target, and gives the alt allele to exactly those samples. Placing the
mutation on a real genealogical branch is what keeps the injected variant in
honest linkage disequilibrium with the background, and makes Hardy-Weinberg
hold once haplotypes are paired at random.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import msprime
import numpy as np

SUPER_POPS = ["AFR", "AMR", "EAS", "EUR", "SAS"]


@dataclass(frozen=True)
class Region:
    """A genomic interval, 1-based inclusive in GRCh38 coordinates."""

    chrom: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    def contains(self, position: int) -> bool:
        return self.start <= position <= self.end

    def offset(self, position: int) -> int:
        """Convert a GRCh38 position to a 0-based offset inside the region."""
        return position - self.start


def region_from_catalog(catalog, flank: int = 20_000) -> Region:
    """Span every catalog variant, plus a flank for recombination context."""
    positions = [int(p) for p in catalog["grch38_pos"].dropna()]
    if not positions:
        raise ValueError("variant catalog has no GRCh38 positions")
    return Region("chr1", min(positions) - flank, max(positions) + flank)


# ----------------------------------------------------------------------
# Demography
# ----------------------------------------------------------------------


def build_demography(spec: dict[str, Any]) -> tuple[msprime.Demography, dict[str, str]]:
    """Build the five-population demography from config.

    The AFR / EUR / EAS backbone comes from Jouganous et al. 2017 as packaged by
    stdpopsim (``OutOfAfrica_4J17``). That model has no South Asian or admixed
    American population, so those two are added here from parameters supplied in
    ``config.yaml``. Every added parameter is tagged in the returned provenance
    map, and the ones marked ``ASSUMPTION`` are exactly that: they shape the
    background genome, not the catalog-variant frequencies that Layer A is
    tested on, and the realised Fst is reported in the QC report so a reader can
    judge them.
    """
    import stdpopsim

    species = stdpopsim.get_species("HomSap")
    base_model = species.get_demographic_model(spec["base_model"])
    demography = base_model.model.copy()

    provenance = {
        "AFR": f"{spec['base_model']} population YRI (Jouganous et al. 2017)",
        "EUR": f"{spec['base_model']} population CEU (Jouganous et al. 2017)",
        "EAS": f"{spec['base_model']} population CHB (Jouganous et al. 2017)",
    }

    # --- South Asia ------------------------------------------------
    #
    # Added with a mass migration rather than ``add_population_split``. A
    # population split marks the ancestral population inactive at the present,
    # which would make the base model's own CEU samples unsamplable. A mass
    # migration moves every SAS lineage into CEU at the split time and leaves
    # CEU otherwise untouched, matching how the base model expresses its own
    # splits.
    sas = spec["added_populations"]["SAS"]
    demography.add_population(
        name="SAS",
        initial_size=float(sas["initial_size"]),
        growth_rate=float(sas.get("growth_rate", 0.0)),
        description="South Asian; added to the base model",
    )
    demography.add_mass_migration(
        time=float(sas["split_time_generations"]),
        source="SAS",
        dest=sas["splits_from"],
        proportion=1.0,
    )
    provenance["SAS"] = (
        f"ASSUMPTION: split from {sas['splits_from']} at "
        f"{sas['split_time_generations']} generations, Ne={sas['initial_size']}. "
        f"{sas.get('rationale', '')}"
    )

    # --- admixed American -------------------------------------------
    amr = spec["added_populations"]["AMR"]
    demography.add_population(
        name="AMR",
        initial_size=float(amr["initial_size"]),
        growth_rate=float(amr.get("growth_rate", 0.0)),
        description="Admixed American; added to the base model",
    )
    # Backwards in time, each source takes a conditional share of whatever
    # lineages are still in AMR, so the realised ancestry fractions match the
    # configured proportions. The last source takes everything remaining.
    admixture_time = float(amr["admixture_time_generations"])
    proportions = dict(amr["ancestral_proportions"])
    remaining = 1.0
    sources = list(proportions)
    for i, source_pop in enumerate(sources):
        share = float(proportions[source_pop])
        is_last = i == len(sources) - 1
        conditional = 1.0 if is_last else min(share / remaining, 1.0)
        demography.add_mass_migration(
            time=admixture_time,
            source="AMR",
            dest=source_pop,
            proportion=conditional,
        )
        remaining -= share
    provenance["AMR"] = (
        f"ASSUMPTION: admixture at {amr['admixture_time_generations']} generations "
        f"from {amr['ancestral_proportions']}, Ne={amr['initial_size']}. "
        f"{amr.get('rationale', '')}"
    )

    demography.sort_events()
    return demography, provenance


POP_TO_MODEL = {"AFR": "YRI", "EUR": "CEU", "EAS": "CHB", "SAS": "SAS", "AMR": "AMR"}


# ----------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------


def simulate_background(
    demography: msprime.Demography,
    samples_per_pop: dict[str, int],
    region: Region,
    seed: int,
    recombination_map: str | None = "HapMapII_GRCh38",
    mutation_rate: float | None = None,
) -> tuple[Any, dict[str, np.ndarray]]:
    """Coalescent simulation of the region, returning the tree sequence.

    ``samples_per_pop`` is counted in diploid individuals. The second return
    value maps each super population to the haplotype (sample node) indices
    belonging to it.
    """
    import stdpopsim

    species = stdpopsim.get_species("HomSap")

    rate_map = None
    if recombination_map:
        genetic_map = species.get_genetic_map(recombination_map)
        contig_map = genetic_map.get_chromosome_map("chr1")
        rate_map = contig_map.slice(
            left=region.start - 1, right=region.end, trim=True
        )
    if mutation_rate is None:
        mutation_rate = species.genome.mean_mutation_rate

    samples = [
        msprime.SampleSet(count, population=POP_TO_MODEL[pop], ploidy=2)
        for pop, count in samples_per_pop.items()
        if count > 0
    ]

    # msprime takes a RateMap through recombination_rate; the map then also
    # fixes the sequence length, so it must not be passed twice.
    if rate_map is not None:
        ts = msprime.sim_ancestry(
            samples=samples,
            demography=demography,
            recombination_rate=rate_map,
            random_seed=seed,
            discrete_genome=True,
        )
    else:
        ts = msprime.sim_ancestry(
            samples=samples,
            demography=demography,
            recombination_rate=1e-8,
            sequence_length=region.length,
            random_seed=seed,
            discrete_genome=True,
        )
    ts = msprime.sim_mutations(ts, rate=mutation_rate, random_seed=seed + 1)

    # Map each super population to its haplotype indices.
    name_to_pop = {POP_TO_MODEL[p]: p for p in samples_per_pop}
    hap_index: dict[str, list[int]] = {pop: [] for pop in samples_per_pop}
    for sample_id in ts.samples():
        model_pop = ts.population(ts.node(sample_id).population).metadata.get("name")
        super_pop = name_to_pop.get(model_pop)
        if super_pop is not None:
            hap_index[super_pop].append(int(sample_id))
    return ts, {k: np.array(v, dtype=np.int64) for k, v in hap_index.items()}


# ----------------------------------------------------------------------
# Injection
# ----------------------------------------------------------------------


def counts_by_population(
    tree, sample_pop_code: np.ndarray, n_pops: int, num_nodes: int
) -> np.ndarray:
    """Samples of each population beneath every node, in one postorder pass.

    ``sample_pop_code`` is indexed by node id and holds the population code for
    sample nodes and ``-1`` everywhere else. Doing all populations together
    keeps injection to one traversal per local tree rather than one per
    population.
    """
    counts = np.zeros((num_nodes, n_pops), dtype=np.int32)
    for node in tree.nodes(order="postorder"):
        code = sample_pop_code[node]
        if code >= 0:
            counts[node, code] += 1
        for child in tree.children(node):
            counts[node] += counts[child]
    return counts


def inject_variant(
    tree,
    pop_samples: np.ndarray,
    target_count: int,
    rng: np.random.Generator,
    node_counts: np.ndarray,
    pop_code: int,
    pop_mask: np.ndarray,
) -> np.ndarray:
    """Pick the haplotypes that carry one injected variant in one population.

    Returns the subset of ``pop_samples`` carrying the alt allele. The clade
    whose size is closest to ``target_count`` is used, then the set is topped up
    or trimmed at random so the realised count matches the target exactly.
    """
    n_pop = len(pop_samples)
    target_count = int(np.clip(target_count, 0, n_pop))
    if target_count == 0:
        return np.array([], dtype=np.int64)
    if target_count == n_pop:
        return pop_samples.copy()

    column = node_counts[:, pop_code]
    nonzero = np.flatnonzero(column > 0)
    if nonzero.size == 0:
        return np.sort(rng.choice(pop_samples, size=target_count, replace=False))

    deltas = np.abs(column[nonzero] - target_count)
    best = nonzero[deltas == deltas.min()]
    node = int(best[rng.integers(best.size)])

    samples = np.fromiter(tree.samples(node), dtype=np.int64)
    carriers_arr = np.sort(samples[pop_mask[samples]])

    if len(carriers_arr) > target_count:
        carriers_arr = np.sort(
            rng.choice(carriers_arr, size=target_count, replace=False)
        )
    elif len(carriers_arr) < target_count:
        remaining = np.setdiff1d(pop_samples, carriers_arr)
        extra = rng.choice(
            remaining, size=target_count - len(carriers_arr), replace=False
        )
        carriers_arr = np.sort(np.concatenate([carriers_arr, extra]))
    return carriers_arr


def inject_catalog(
    ts,
    hap_index: dict[str, np.ndarray],
    variants: list[dict[str, Any]],
    targets: dict[str, dict[str, float]],
    region: Region,
    seed: int,
    haplotype_groups: dict[str, list[str]] | None = None,
) -> dict[str, np.ndarray]:
    """Inject every catalog variant at its per-population target frequency.

    ``targets[variant_id][population]`` is the target alternate allele
    frequency. Returns a mapping of variant id to a boolean array over all
    haplotypes in the tree sequence.

    ``haplotype_groups`` names sets of variants that travel together on one
    haplotype, such as the two variants defining HapB3. Injecting them
    independently would scatter them across different chromosomes and destroy
    the linkage that makes them one allele, so each group is placed once and
    every member inherits the same carriers.
    """
    rng = np.random.default_rng(seed)
    n_haplotypes = ts.num_samples
    out: dict[str, np.ndarray] = {}

    pops = sorted(hap_index)
    pop_code = {pop: i for i, pop in enumerate(pops)}
    sample_pop_code = np.full(ts.num_nodes, -1, dtype=np.int8)
    pop_masks: dict[str, np.ndarray] = {}
    for pop in pops:
        sample_pop_code[hap_index[pop]] = pop_code[pop]
        mask = np.zeros(ts.num_nodes, dtype=bool)
        mask[hap_index[pop]] = True
        pop_masks[pop] = mask

    # Map each variant to the haplotype group it belongs to, if any. The group
    # is injected once, at the position of its first member.
    group_of: dict[str, str] = {}
    for group_name, members in (haplotype_groups or {}).items():
        for member in members:
            group_of[member] = group_name
    injected_groups: dict[str, np.ndarray] = {}

    for variant in sorted(variants, key=lambda v: int(v["grch38_pos"])):
        variant_id = str(variant["variant_id"])
        position = int(variant["grch38_pos"])
        if not region.contains(position):
            continue

        group = group_of.get(variant_id)
        if group is not None and group in injected_groups:
            out[variant_id] = injected_groups[group]
            continue

        carriers = np.zeros(n_haplotypes, dtype=bool)
        tree = ts.at(region.offset(position))
        node_counts = counts_by_population(
            tree, sample_pop_code, len(pops), ts.num_nodes
        )
        per_variant_targets = targets.get(variant_id, {})
        for pop in pops:
            pop_samples = hap_index[pop]
            frequency = per_variant_targets.get(pop)
            if frequency is None or not np.isfinite(frequency) or frequency <= 0:
                continue
            target_count = int(round(float(frequency) * len(pop_samples)))
            chosen = inject_variant(
                tree,
                pop_samples,
                target_count,
                rng,
                node_counts,
                pop_code[pop],
                pop_masks[pop],
            )
            carriers[chosen] = True
        out[variant_id] = carriers
        if group is not None:
            injected_groups[group] = carriers
    return out


# ----------------------------------------------------------------------
# Diploid assembly
# ----------------------------------------------------------------------


def pair_haplotypes(
    hap_index: dict[str, np.ndarray], seed: int
) -> dict[str, np.ndarray]:
    """Pair each population's haplotypes into diploid individuals.

    Random pairing within a population is what makes Hardy-Weinberg hold, and
    msprime already emits haplotypes two per simulated individual, so the
    default pairing is kept and only shuffled in order.
    """
    rng = np.random.default_rng(seed)
    out: dict[str, np.ndarray] = {}
    for pop, haplotypes in hap_index.items():
        n_pairs = len(haplotypes) // 2
        pairs = haplotypes[: n_pairs * 2].reshape(n_pairs, 2)
        order = rng.permutation(n_pairs)
        out[pop] = pairs[order]
    return out


def genotype_codes(carriers: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Convert haplotype carrier flags into 0/1/2 diploid dosages."""
    return carriers[pairs[:, 0]].astype(np.int8) + carriers[pairs[:, 1]].astype(np.int8)


GENOTYPE_STRINGS = {0: "0/0", 1: "0/1", 2: "1/1"}
PHASED_STRINGS = {(0, 0): "0|0", (0, 1): "0|1", (1, 0): "1|0", (1, 1): "1|1"}


def background_matrix(
    ts, pairs: np.ndarray, max_sites: int = 4000, min_maf: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    """Dosage matrix of common background sites, for PCA and LD checks."""
    dosages: list[np.ndarray] = []
    positions: list[float] = []
    n_hap = ts.num_samples
    for variant in ts.variants():
        if len(variant.alleles) != 2:
            continue
        genotypes = variant.genotypes
        freq = genotypes.sum() / n_hap
        if freq < min_maf or freq > 1 - min_maf:
            continue
        dosages.append(
            genotypes[pairs[:, 0]].astype(np.int8)
            + genotypes[pairs[:, 1]].astype(np.int8)
        )
        positions.append(variant.site.position)
        if len(dosages) >= max_sites:
            break
    if not dosages:
        return np.zeros((len(pairs), 0), dtype=np.int8), np.array([])
    return np.array(dosages, dtype=np.int8).T, np.array(positions)
