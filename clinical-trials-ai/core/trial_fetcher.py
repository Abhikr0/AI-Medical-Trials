"""ClinicalTrials.gov REST API v2 client and study extraction engine."""

import hashlib
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

import config

logger = logging.getLogger(__name__)

USER_AGENT: str = "clinical-trials-ai/1.0 (prototype)"
DEFAULT_TIMEOUT: float = 30.0
FIELDS_PROJECTION: str = (
    "NCTId,BriefTitle,OfficialTitle,BriefSummary,Condition,EligibilityCriteria,"
    "Phase,OverallStatus,StudyType,LocationCity,LocationState,LocationCountry"
)

PHASE_MAP: Dict[str, str] = {
    "EARLY_PHASE1": "Early Phase 1",
    "PHASE1": "Phase 1",
    "PHASE2": "Phase 2",
    "PHASE3": "Phase 3",
    "PHASE4": "Phase 4",
    "NA": "Not Applicable",
    "NOT_APPLICABLE": "Not Applicable",
}


def parse_phase(phases: Optional[Any]) -> str:
    """Normalize phase lists or strings to human-readable strings.

    Examples:
        ["PHASE1"] -> "Phase 1"
        ["PHASE1", "PHASE2"] -> "Phase 1, Phase 2"
        ["NA"] or None or [] -> "Not Applicable"
        "EARLY_PHASE1" -> "Early Phase 1"
    """
    if phases is None:
        return "Not Applicable"

    if isinstance(phases, str):
        phases = [phases]

    if not isinstance(phases, (list, tuple)):
        return "Not Applicable"

    clean_phases = [str(p).strip() for p in phases if p and str(p).strip()]
    if not clean_phases:
        return "Not Applicable"

    mapped_phases: List[str] = []
    for p in clean_phases:
        upper_p = p.upper()
        if upper_p in PHASE_MAP:
            normalized = PHASE_MAP[upper_p]
        elif p in PHASE_MAP:
            normalized = PHASE_MAP[p]
        else:
            spaced = re.sub(r"([a-zA-Z]+)(\d+)", r"\1 \2", p)
            normalized = spaced.replace("_", " ").title()

        if normalized not in mapped_phases:
            mapped_phases.append(normalized)

    if not mapped_phases or mapped_phases == ["Not Applicable"]:
        return "Not Applicable"

    # If both real phases and 'Not Applicable' exist, discard 'Not Applicable'
    if len(mapped_phases) > 1 and "Not Applicable" in mapped_phases:
        mapped_phases = [m for m in mapped_phases if m != "Not Applicable"]

    return ", ".join(mapped_phases) if mapped_phases else "Not Applicable"


def extract_trial(study: Optional[Dict[str, Any]]) -> Tuple[str, Dict[str, str]]:
    """Extract embeddable text and ChromaDB-compliant primitive metadata from a study.

    Args:
        study: Raw ClinicalTrials.gov study dict matching API v2 schema.

    Returns:
        tuple[str, dict]: (embed_text, metadata) where metadata contains strictly primitive strings.
    """
    if not isinstance(study, dict):
        study = {}

    proto = study.get("protocolSection", {})
    if not isinstance(proto, dict):
        proto = study

    id_m = proto.get("identificationModule", {}) if isinstance(proto, dict) else {}
    st_m = proto.get("statusModule", {}) if isinstance(proto, dict) else {}
    ds_m = proto.get("descriptionModule", {}) if isinstance(proto, dict) else {}
    cd_m = proto.get("conditionsModule", {}) if isinstance(proto, dict) else {}
    dg_m = proto.get("designModule", {}) if isinstance(proto, dict) else {}
    el_m = proto.get("eligibilityModule", {}) if isinstance(proto, dict) else {}
    cl_m = proto.get("contactsLocationsModule", {}) if isinstance(proto, dict) else {}

    # Primary identification fields
    nct_id = (
        (id_m.get("nctId") if isinstance(id_m, dict) else None)
        or study.get("nctId")
        or study.get("nct_id")
        or ""
    )
    brief_title = (
        (id_m.get("briefTitle") if isinstance(id_m, dict) else None)
        or study.get("briefTitle")
        or study.get("title")
        or ""
    )
    official_title = (
        (id_m.get("officialTitle") if isinstance(id_m, dict) else None)
        or study.get("officialTitle")
        or study.get("official_title")
        or ""
    )
    overall_status = (
        (st_m.get("overallStatus") if isinstance(st_m, dict) else None)
        or study.get("overallStatus")
        or study.get("status")
        or ""
    )
    brief_summary = (
        (ds_m.get("briefSummary") if isinstance(ds_m, dict) else None)
        or study.get("briefSummary")
        or study.get("summary")
        or ""
    )
    conditions_raw = (
        (cd_m.get("conditions") if isinstance(cd_m, dict) else None)
        or study.get("conditions")
        or []
    )
    phases_raw = (
        (dg_m.get("phases") if isinstance(dg_m, dict) else None)
        or study.get("phases")
        or []
    )
    study_type = (
        (dg_m.get("studyType") if isinstance(dg_m, dict) else None)
        or study.get("studyType")
        or study.get("study_type")
        or ""
    )
    eligibility_criteria = (
        (el_m.get("eligibilityCriteria") if isinstance(el_m, dict) else None)
        or study.get("eligibilityCriteria")
        or study.get("eligibility")
        or ""
    )
    locations_raw = (
        (cl_m.get("locations") if isinstance(cl_m, dict) else None)
        or study.get("locations")
        or []
    )

    # Fallback deterministic NCT ID if missing
    if not nct_id:
        fallback_hash = hashlib.sha256(
            f"{brief_title}:{official_title}:{brief_summary}".encode("utf-8")
        ).hexdigest()[:8].upper()
        nct_id = f"NCT_{fallback_hash}"

    # Normalize conditions
    if isinstance(conditions_raw, list):
        clean_conds = [str(c).strip() for c in conditions_raw if c and str(c).strip()]
        cond_str = ", ".join(clean_conds)
    elif isinstance(conditions_raw, str):
        cond_str = conditions_raw.strip()
    else:
        cond_str = ""

    # Parse human-readable phase
    parsed_phase = parse_phase(phases_raw)

    # Summarize site locations safely
    if not isinstance(locations_raw, list):
        locations_raw = []

    loc_strs: List[str] = []
    for loc in locations_raw[:5]:
        if isinstance(loc, dict):
            city = loc.get("city")
            state = loc.get("state")
            country = loc.get("country")
            parts = [str(p).strip() for p in (city, state, country) if p and str(p).strip()]
            loc_str = ", ".join(parts)
            if loc_str and loc_str not in loc_strs:
                loc_strs.append(loc_str)
        elif isinstance(loc, str) and loc.strip():
            if loc.strip() not in loc_strs:
                loc_strs.append(loc.strip())

    if loc_strs:
        locations_summary = "; ".join(loc_strs)
        if len(locations_raw) > 5:
            locations_summary += f" (+{len(locations_raw) - 5} more)"
    else:
        locations_summary = "Location not specified"

    # Construct embeddable text
    # Prioritizes Title and Conditions first (MiniLM 256-token limit), followed by truncated summary & eligibility
    text_components: List[str] = []
    if brief_title and str(brief_title).strip():
        text_components.append(f"Title: {str(brief_title).strip()}")
    if cond_str:
        text_components.append(f"Conditions: {cond_str}")
    if brief_summary and str(brief_summary).strip():
        text_components.append(f"Summary: {str(brief_summary).strip()[:500]}")
    if eligibility_criteria and str(eligibility_criteria).strip():
        text_components.append(f"Eligibility: {str(eligibility_criteria).strip()[:400]}")

    embed_text = " | ".join(text_components)

    # Strictly primitive metadata for ChromaDB compliance
    metadata: Dict[str, str] = {
        "nct_id": str(nct_id).strip(),
        "title": str(brief_title or "").strip()[:200],
        "official_title": (
            str(official_title or brief_title or "").strip()[:200]
        ),
        "conditions": cond_str[:200],
        "phase": parsed_phase,
        "status": str(overall_status or "").strip(),
        "study_type": str(study_type or "").strip(),
        "locations": locations_summary[:200],
        "url": f"https://clinicaltrials.gov/study/{str(nct_id).strip()}",
    }

    return embed_text, metadata


def fetch_trials(
    conditions: List[str],
    page_size: int = 200,
    status: str = "RECRUITING",
    max_retries: int = 3,
    backoff_factor: float = 0.5,
) -> List[Dict[str, Any]]:
    """Fetch clinical trials from ClinicalTrials.gov REST API v2 matching conditions and status.

    Args:
        conditions: List of disease/condition search terms.
        page_size: Number of trials per request (clamped between 1 and 1000).
        status: Overall study recruitment status filter (e.g. 'RECRUITING', 'ALL').
        max_retries: Maximum exponential backoff retries on 429/503.
        backoff_factor: Initial backoff sleep multiplier in seconds.

    Returns:
        List of raw study dictionary objects from the API.

    Raises:
        httpx.HTTPStatusError: On non-retryable 4xx/5xx responses or exhausted retries.
        httpx.RequestError: On network or transport connection failures.
    """
    base_url = getattr(config, "CLINICALTRIALS_BASE", "https://clinicaltrials.gov/api/v2")
    endpoint = f"{base_url.rstrip('/')}/studies"

    clamped_page_size = max(1, min(1000, int(page_size)))

    params: Dict[str, Any] = {
        "pageSize": clamped_page_size,
        "format": "json",
        "fields": FIELDS_PROJECTION,
    }

    # Format query conditions
    if conditions:
        clean_conditions = [str(c).strip() for c in conditions if c and str(c).strip()]
        if clean_conditions:
            params["query.cond"] = " OR ".join(
                f'"{c}"' if " " in c else c for c in clean_conditions
            )

    # Format status filter
    if status and str(status).strip().upper() != "ALL":
        params["filter.overallStatus"] = str(status).strip()

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }

    for attempt in range(max_retries + 1):
        try:
            with httpx.Client(timeout=DEFAULT_TIMEOUT, headers=headers) as client:
                response = client.get(endpoint, params=params)

                if response.status_code in (429, 503):
                    if attempt < max_retries:
                        retry_after = response.headers.get("Retry-After")
                        if retry_after and retry_after.isdigit():
                            sleep_time = float(retry_after)
                        else:
                            sleep_time = backoff_factor * (2 ** attempt)
                        logger.warning(
                            f"HTTP {response.status_code} received from ClinicalTrials API. "
                            f"Retrying in {sleep_time:.2f}s (attempt {attempt + 1}/{max_retries})..."
                        )
                        time.sleep(sleep_time)
                        continue
                    else:
                        response.raise_for_status()

                response.raise_for_status()
                data = response.json()
                return data.get("studies", [])

        except httpx.HTTPStatusError as e:
            if e.response.status_code in (429, 503) and attempt < max_retries:
                sleep_time = backoff_factor * (2 ** attempt)
                logger.warning(
                    f"HTTPStatusError {e.response.status_code}. Retrying in {sleep_time:.2f}s..."
                )
                time.sleep(sleep_time)
                continue
            logger.error(f"HTTP status error while fetching trials: {e}")
            raise
        except httpx.RequestError as e:
            if attempt < max_retries:
                sleep_time = backoff_factor * (2 ** attempt)
                logger.warning(f"RequestError {e}. Retrying in {sleep_time:.2f}s...")
                time.sleep(sleep_time)
                continue
            logger.error(f"Network error while fetching trials: {e}")
            raise

    return []


def preload_standard_trials(
    categories: Optional[List[str]] = None,
    page_size: int = 200,
) -> List[Dict[str, Any]]:
    """Preload recruiting clinical trials for broad condition categories.

    Queries broad medical categories (cancer, diabetes, heart disease, neurological)
    and deduplicates results by NCT ID.

    Args:
        categories: Optional list of categories. Defaults to standard 4 broad categories.
        page_size: Target total number of studies across categories.

    Returns:
        List of distinct study dictionary objects.
    """
    if categories is None:
        categories = ["cancer", "diabetes", "heart disease", "neurological"]

    if not categories:
        return []

    num_cats = len(categories)
    per_cat_page_size = max(1, page_size // num_cats)

    seen_ncts = set()
    aggregated_studies: List[Dict[str, Any]] = []

    for cat in categories:
        try:
            studies = fetch_trials(
                conditions=[cat], page_size=per_cat_page_size, status="RECRUITING"
            )
        except Exception as e:
            logger.warning(f"Failed to fetch trials for category '{cat}': {e}")
            continue

        for study in studies:
            nct_id = None
            if isinstance(study, dict):
                proto = study.get("protocolSection", {})
                if isinstance(proto, dict):
                    id_m = proto.get("identificationModule", {})
                    if isinstance(id_m, dict):
                        nct_id = id_m.get("nctId")
                if not nct_id:
                    nct_id = study.get("nctId") or study.get("nct_id")

            if nct_id:
                if nct_id not in seen_ncts:
                    seen_ncts.add(nct_id)
                    aggregated_studies.append(study)
            else:
                aggregated_studies.append(study)

    return aggregated_studies
