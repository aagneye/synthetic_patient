"""HTTP source clients with on-disk caching.

SYNTHETIC. In silico cohort validation. Not clinical evidence.

Every external value used by this dataset is fetched through this module and
cached as raw JSON under ``reference/raw/``.  Nothing here invents a value: if a
source has no answer, the caller receives ``None`` and is expected to record the
gap in ``qc/open_questions.md``.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

USER_AGENT = "synthpgx/1.0 (Anukriti AI; synthetic PGx dataset build)"

CPIC_API = "https://api.cpicpgx.org/v1/"
ENSEMBL_API = "https://rest.ensembl.org/"
GNOMAD_API = "https://gnomad.broadinstitute.org/api"
EUTILS_API = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


class SourceError(RuntimeError):
    """Raised when a source is reachable but returns an error payload."""


class Fetcher:
    """Cached HTTP client.

    The cache makes the build reproducible and keeps re-runs off the public
    APIs.  Delete ``reference/raw/`` to force a refresh.
    """

    def __init__(self, cache_dir: Path, offline: bool = False, pause: float = 0.12):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.offline = offline
        self.pause = pause
        self.stats = {"hit": 0, "miss": 0, "fail": 0}

    # ------------------------------------------------------------------
    def _cache_path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode()).hexdigest()[:20]
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)[:70]
        return self.cache_dir / f"{safe}.{digest}.json"

    def _load(self, key: str) -> Any | None:
        path = self._cache_path(key)
        if path.exists():
            self.stats["hit"] += 1
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def _store(self, key: str, value: Any) -> None:
        self._cache_path(key).write_text(
            json.dumps(value, indent=1, sort_keys=True), encoding="utf-8"
        )

    # ------------------------------------------------------------------
    def _request(self, url: str, data: bytes | None = None, tries: int = 4) -> Any:
        headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
        if data is not None:
            headers["Content-Type"] = "application/json"
        last: Exception | None = None
        for attempt in range(tries):
            try:
                req = urllib.request.Request(url, data=data, headers=headers)
                with urllib.request.urlopen(req, timeout=120) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                time.sleep(self.pause)
                return payload
            except urllib.error.HTTPError as exc:  # pragma: no cover - network
                last = exc
                if exc.code in (400, 401, 403, 404):
                    raise SourceError(f"HTTP {exc.code} for {url}") from exc
                time.sleep(1.5 * (attempt + 1))
            except Exception as exc:  # pragma: no cover - network
                last = exc
                time.sleep(1.5 * (attempt + 1))
        raise SourceError(f"request failed for {url}: {last!r}")

    def get_json(self, key: str, url: str) -> Any:
        cached = self._load(key)
        if cached is not None:
            return cached
        if self.offline:
            raise SourceError(f"offline and no cache for {key}")
        self.stats["miss"] += 1
        value = self._request(url)
        self._store(key, value)
        return value

    def post_graphql(self, key: str, query: str) -> Any:
        cached = self._load(key)
        if cached is not None:
            return cached
        if self.offline:
            raise SourceError(f"offline and no cache for {key}")
        self.stats["miss"] += 1
        value = self._request(GNOMAD_API, data=json.dumps({"query": query}).encode())
        self._store(key, value)
        return value

    def try_json(self, key: str, url: str) -> Any | None:
        try:
            return self.get_json(key, url)
        except SourceError:
            self.stats["fail"] += 1
            return None


# ----------------------------------------------------------------------
# CPIC
# ----------------------------------------------------------------------

CPIC_QUERIES: dict[str, str] = {
    "gene": "gene?symbol=eq.DPYD",
    "allele": "allele?genesymbol=eq.DPYD&order=name",
    "allele_definition": "allele_definition?genesymbol=eq.DPYD&order=name",
    "sequence_location": "sequence_location?genesymbol=eq.DPYD&order=position",
    "drug": "drug?name=in.(fluorouracil,capecitabine)",
}


def fetch_cpic(fetcher: Fetcher) -> dict[str, Any]:
    """Pull the DPYD slice of the CPIC database.

    Returns raw rows keyed by endpoint.  These rows are the authority for
    Layer A: allele function, activity value, diplotype -> phenotype, and the
    fluoropyrimidine recommendations.
    """
    out: dict[str, Any] = {}
    for name, query in CPIC_QUERIES.items():
        out[name] = fetcher.get_json(f"cpic_{name}", CPIC_API + query)

    # Diplotype table is large; fetch it whole for the gene.
    out["diplotype"] = fetcher.get_json(
        "cpic_diplotype", CPIC_API + "diplotype?genesymbol=eq.DPYD"
    )

    # allele_location_value is keyed by allele definition id, not gene.
    def_ids = [str(d["id"]) for d in out["allele_definition"]]
    values: list[dict[str, Any]] = []
    for i in range(0, len(def_ids), 40):
        chunk = ",".join(def_ids[i : i + 40])
        values.extend(
            fetcher.get_json(
                f"cpic_alv_{i}",
                CPIC_API + f"allele_location_value?alleledefinitionid=in.({chunk})",
            )
        )
    out["allele_location_value"] = values

    drug_ids = ",".join(d["drugid"] for d in out["drug"])
    out["recommendation"] = fetcher.get_json(
        "cpic_recommendation",
        CPIC_API
        + "recommendation?drugid=in.("
        + urllib.parse.quote(drug_ids, safe=",:")
        + ")&select=drugid,phenotypes,activityscore,drugrecommendation,implications,"
        "classification,lookupkey,comments",
    )
    return out


# ----------------------------------------------------------------------
# Ensembl
# ----------------------------------------------------------------------


def ensembl_recode(fetcher: Fetcher, hgvs: str) -> dict[str, Any] | None:
    """Resolve an HGVS string to rsID, GRCh38 placement and protein change.

    This is the rsID *verification* step.  The caller supplies the coding HGVS
    and the protein change it expects; the returned record carries what Ensembl
    actually says, so a mismatch is detectable rather than silently accepted.
    """
    url = (
        ENSEMBL_API
        + "variant_recoder/human/"
        + urllib.parse.quote(hgvs)
        + "?content-type=application/json"
    )
    payload = fetcher.try_json(f"ensembl_recode_{hgvs}", url)
    if not payload:
        return None
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        for _allele, rec in entry.items():
            if isinstance(rec, dict) and rec.get("id"):
                return rec
    return None


def ensembl_1kg_frequencies(fetcher: Fetcher, rsid: str) -> dict[str, float]:
    """1000 Genomes phase 3 sub-population allele frequencies for an rsID.

    Supplemental only.  gnomAD is the primary frequency source (spec 6.4.3,
    because 1000 Genomes samples are also inside gnomAD and must not be
    double-counted).
    """
    url = (
        ENSEMBL_API
        + f"variation/human/{rsid}?pops=1&content-type=application/json"
    )
    payload = fetcher.try_json(f"ensembl_pop_{rsid}", url)
    if not payload:
        return {}
    minor = payload.get("minor_allele")
    out: dict[str, float] = {}
    for pop in payload.get("populations", []):
        name = pop.get("population", "")
        if not name.startswith("1000GENOMES:phase_3:"):
            continue
        code = name.rsplit(":", 1)[-1]
        if code not in {"GIH", "ITU", "PJL", "BEB", "STU"}:
            continue
        # Keep the frequency of the non-reference (minor) allele only.
        if minor and pop.get("allele") != minor:
            continue
        out[code] = float(pop.get("frequency", 0.0))
    return out


# ----------------------------------------------------------------------
# gnomAD
# ----------------------------------------------------------------------

#: gnomAD ancestry code -> Anukriti closed-enum super population.
GNOMAD_POP_MAP = {
    "afr": "AFR",
    "amr": "AMR",
    "eas": "EAS",
    "nfe": "EUR",
    "sas": "SAS",
}

_VARIANT_QUERY = """
{
  variant(variantId: "%s", dataset: %s) {
    variant_id
    rsids
    genome { ac an populations { id ac an } }
    exome  { ac an populations { id ac an } }
  }
}
"""


def gnomad_variant(
    fetcher: Fetcher, variant_id: str, dataset: str = "gnomad_r3"
) -> dict[str, Any] | None:
    """Per-ancestry allele frequencies for one variant.

    ``dataset`` is ``gnomad_r3`` (v3.1.2) by default because spec 6.4.3 names
    v3.1.2 as the primary frequency source.
    """
    key = f"gnomad_{dataset}_{variant_id}"
    try:
        payload = fetcher.post_graphql(key, _VARIANT_QUERY % (variant_id, dataset))
    except SourceError:
        fetcher.stats["fail"] += 1
        return None
    variant = (payload or {}).get("data", {}).get("variant")
    if not variant:
        return None
    return variant


def gnomad_ancestry_af(variant: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Collapse a gnomAD variant record into AF per Anukriti super population.

    Exome and genome callsets are summed (independent sample sets within a
    gnomAD release), and the sex-split ``*_XX`` / ``*_XY`` entries are skipped
    so samples are counted once.
    """
    totals: dict[str, list[int]] = {v: [0, 0] for v in GNOMAD_POP_MAP.values()}
    for callset_name in ("genome", "exome"):
        callset = variant.get(callset_name)
        if not callset:
            continue
        for pop in callset.get("populations", []):
            pid = pop.get("id", "")
            if "_" in pid:  # skip XX / XY sex strata
                continue
            super_pop = GNOMAD_POP_MAP.get(pid)
            if not super_pop:
                continue
            totals[super_pop][0] += int(pop.get("ac") or 0)
            totals[super_pop][1] += int(pop.get("an") or 0)
    out: dict[str, dict[str, Any]] = {}
    for super_pop, (ac, an) in totals.items():
        out[super_pop] = {
            "ac": ac,
            "an": an,
            "af": (ac / an) if an else None,
        }
    return out


# ----------------------------------------------------------------------
# PubMed
# ----------------------------------------------------------------------


def verify_pmids(fetcher: Fetcher, pmids: list[str]) -> dict[str, dict[str, str]]:
    """Confirm each PMID resolves, and return first author / journal / title."""
    if not pmids:
        return {}
    url = (
        EUTILS_API
        + "esummary.fcgi?db=pubmed&retmode=json&id="
        + ",".join(sorted(set(pmids)))
    )
    payload = fetcher.try_json("pubmed_" + hashlib.sha1(url.encode()).hexdigest()[:10], url)
    if not payload:
        return {}
    result = payload.get("result", {})
    out: dict[str, dict[str, str]] = {}
    for uid in result.get("uids", []):
        entry = result[uid]
        authors = entry.get("authors") or []
        out[uid] = {
            "first_author": authors[0]["name"] if authors else "",
            "journal": entry.get("source", ""),
            "pubdate": entry.get("pubdate", ""),
            "title": entry.get("title", ""),
        }
    return out
