"""Comprehensive unit test suite for core/trial_fetcher.py."""

from unittest.mock import MagicMock, call, patch
import httpx
import pytest

from core.trial_fetcher import (
    DEFAULT_TIMEOUT,
    FIELDS_PROJECTION,
    USER_AGENT,
    extract_trial,
    fetch_trials,
    parse_phase,
    preload_standard_trials,
)
from tests.conftest import make_mock_study


# ============================================================================
# 1. Phase Normalization Unit Tests (parse_phase)
# ============================================================================

class TestParsePhase:
    """Test parse_phase under all standard and boundary conditions."""

    def test_single_standard_phases(self):
        assert parse_phase(["EARLY_PHASE1"]) == "Early Phase 1"
        assert parse_phase(["PHASE1"]) == "Phase 1"
        assert parse_phase(["PHASE2"]) == "Phase 2"
        assert parse_phase(["PHASE3"]) == "Phase 3"
        assert parse_phase(["PHASE4"]) == "Phase 4"

    def test_multi_phase_combinations(self):
        assert parse_phase(["PHASE1", "PHASE2"]) == "Phase 1, Phase 2"
        assert parse_phase(["PHASE2", "PHASE3"]) == "Phase 2, Phase 3"
        assert parse_phase(["EARLY_PHASE1", "PHASE1"]) == "Early Phase 1, Phase 1"

    def test_none_and_empty_inputs(self):
        assert parse_phase(None) == "Not Applicable"
        assert parse_phase([]) == "Not Applicable"
        assert parse_phase([""]) == "Not Applicable"
        assert parse_phase(["   "]) == "Not Applicable"
        assert parse_phase(123) == "Not Applicable"

    def test_na_and_not_applicable_variants(self):
        assert parse_phase(["NA"]) == "Not Applicable"
        assert parse_phase(["NOT_APPLICABLE"]) == "Not Applicable"
        assert parse_phase(["NA", "NA"]) == "Not Applicable"

    def test_string_instead_of_list(self):
        assert parse_phase("PHASE1") == "Phase 1"
        assert parse_phase("EARLY_PHASE1") == "Early Phase 1"
        assert parse_phase("NA") == "Not Applicable"
        assert parse_phase("") == "Not Applicable"

    def test_case_insensitivity(self):
        assert parse_phase(["phase1"]) == "Phase 1"
        assert parse_phase(["phase2", "Phase3"]) == "Phase 2, Phase 3"
        assert parse_phase(["early_phase1"]) == "Early Phase 1"

    def test_deduplication_preserving_order(self):
        assert parse_phase(["PHASE1", "PHASE1"]) == "Phase 1"
        assert parse_phase(["PHASE2", "PHASE1", "PHASE2"]) == "Phase 2, Phase 1"

    def test_mixed_phase_and_na(self):
        assert parse_phase(["PHASE1", "NA"]) == "Phase 1"
        assert parse_phase(["NA", "PHASE3"]) == "Phase 3"

    def test_uncommon_phase_fallback(self):
        assert parse_phase(["PHASE5"]) == "Phase 5"
        assert parse_phase(["POST_MARKETING"]) == "Post Marketing"


# ============================================================================
# 2. Trial Extraction Unit Tests (extract_trial)
# ============================================================================

class TestExtractTrial:
    """Test extract_trial text formatting and metadata serialization."""

    def test_complete_study_extraction(self, sample_study_payload):
        embed_text, metadata = extract_trial(sample_study_payload)

        # embed_text verification
        assert "Title: Immunotherapy Trial for Melanoma" in embed_text
        assert "Conditions: Melanoma, Skin Cancer" in embed_text
        assert "Summary:" in embed_text
        assert "Eligibility:" in embed_text

        # Ordering: Title and Conditions come first
        assert embed_text.index("Title:") < embed_text.index("Conditions:")
        assert embed_text.index("Conditions:") < embed_text.index("Summary:")
        assert embed_text.index("Summary:") < embed_text.index("Eligibility:")

        # Metadata primitive verification
        assert metadata["nct_id"] == "NCT00000001"
        assert metadata["title"] == "Immunotherapy Trial for Melanoma"
        assert metadata["official_title"] == "Phase 3 Study of Pembrolizumab"
        assert "Melanoma" in metadata["conditions"]
        assert metadata["phase"] == "Phase 2"
        assert metadata["status"] == "RECRUITING"
        assert metadata["study_type"] == "INTERVENTIONAL"
        assert "Boston" in metadata["locations"]
        assert metadata["url"] == "https://clinicaltrials.gov/study/NCT00000001"

        # Strictly primitive types
        for k, v in metadata.items():
            assert isinstance(v, str), f"Metadata field '{k}' is not str: {type(v)}"

    def test_missing_identification_generates_deterministic_nct_id(self):
        empty_id_study = {
            "protocolSection": {
                "identificationModule": {},
                "descriptionModule": {"briefSummary": "Some summary."},
            }
        }
        embed_text, metadata = extract_trial(empty_id_study)
        assert metadata["nct_id"].startswith("NCT_")
        assert len(metadata["nct_id"]) > 4
        # Same study yields identical deterministic hash
        _, metadata2 = extract_trial(empty_id_study)
        assert metadata["nct_id"] == metadata2["nct_id"]

    def test_official_title_fallback_to_brief_title(self):
        study = make_mock_study(
            brief_title="Brief Only Title",
            official_title="",
        )
        # Clear officialTitle explicitly
        study["protocolSection"]["identificationModule"]["officialTitle"] = ""
        _, metadata = extract_trial(study)
        assert metadata["official_title"] == "Brief Only Title"

    def test_empty_and_null_study_handling(self):
        embed_text, metadata = extract_trial({})
        assert embed_text == ""
        assert isinstance(metadata, dict)
        assert metadata["locations"] == "Location not specified"
        assert metadata["phase"] == "Not Applicable"
        assert metadata["status"] == ""
        assert metadata["conditions"] == ""

        embed_text_none, metadata_none = extract_trial(None)
        assert embed_text_none == ""
        assert isinstance(metadata_none, dict)

    def test_locations_summarization_and_truncation(self):
        # 3 locations
        study_3_locs = make_mock_study(
            locations=[
                {"city": "Boston", "state": "MA", "country": "United States"},
                {"city": "New York", "state": "NY", "country": "United States"},
                {"city": "Toronto", "state": "ON", "country": "Canada"},
            ]
        )
        _, meta_3 = extract_trial(study_3_locs)
        assert "Boston, MA, United States" in meta_3["locations"]
        assert "New York, NY, United States" in meta_3["locations"]
        assert "Toronto, ON, Canada" in meta_3["locations"]
        assert "(+" not in meta_3["locations"]

        # > 5 locations
        study_7_locs = make_mock_study(
            locations=[
                {"city": f"City{i}", "state": "ST", "country": "USA"}
                for i in range(7)
            ]
        )
        _, meta_7 = extract_trial(study_7_locs)
        assert "(+2 more)" in meta_7["locations"]

    def test_locations_with_sparse_missing_components(self):
        study_sparse = make_mock_study(
            locations=[
                {"city": "Paris", "country": "France"},
                {"country": "Germany"},
                {"city": "Tokyo"},
            ]
        )
        _, meta = extract_trial(study_sparse)
        assert "Paris, France" in meta["locations"]
        assert "Germany" in meta["locations"]
        assert "Tokyo" in meta["locations"]

    def test_locations_empty_defaults_to_specified(self):
        study_no_locs = make_mock_study(locations=[])
        _, meta = extract_trial(study_no_locs)
        assert meta["locations"] == "Location not specified"

    def test_text_and_metadata_length_bounds(self):
        huge_summary = "X" * 1500
        huge_eligibility = "Y" * 2000
        huge_title = "Z" * 500
        huge_conditions = [f"Condition_{i}_{'W'*50}" for i in range(10)]

        huge_study = make_mock_study(
            brief_title=huge_title,
            official_title=huge_title,
            conditions=huge_conditions,
            brief_summary=huge_summary,
            eligibility_criteria=huge_eligibility,
        )

        embed_text, metadata = extract_trial(huge_study)

        # Summary and eligibility bounded in embed_text
        assert f"Summary: {'X'*500}" in embed_text
        assert f"Summary: {'X'*501}" not in embed_text
        assert f"Eligibility: {'Y'*400}" in embed_text
        assert f"Eligibility: {'Y'*401}" not in embed_text

        # Metadata fields bounded to 200 characters
        assert len(metadata["title"]) <= 200
        assert len(metadata["official_title"]) <= 200
        assert len(metadata["conditions"]) <= 200
        assert len(metadata["locations"]) <= 200


# ============================================================================
# 3. Trial Fetching Unit Tests (fetch_trials)
# ============================================================================

class TestFetchTrials:
    """Test fetch_trials HTTP query construction, pagination, and error handling."""

    def test_query_cond_construction_with_spaces_and_quotes(self):
        captured_params = {}

        def mock_get(url, params=None, **kwargs):
            captured_params.update(params or {})
            resp = MagicMock(spec=httpx.Response)
            resp.status_code = 200
            resp.json.return_value = {"studies": []}
            resp.raise_for_status = MagicMock()
            return resp

        with patch("httpx.Client.get", side_effect=mock_get):
            fetch_trials(["asthma", "heart failure", "lung cancer"])

        expected_cond = 'asthma OR "heart failure" OR "lung cancer"'
        assert captured_params.get("query.cond") == expected_cond

    def test_empty_conditions_omits_query_cond(self):
        captured_params = {}

        def mock_get(url, params=None, **kwargs):
            captured_params.update(params or {})
            resp = MagicMock(spec=httpx.Response)
            resp.status_code = 200
            resp.json.return_value = {"studies": []}
            resp.raise_for_status = MagicMock()
            return resp

        with patch("httpx.Client.get", side_effect=mock_get):
            fetch_trials([])

        assert "query.cond" not in captured_params

    def test_page_size_clamping(self):
        captured_sizes = []

        def mock_get(url, params=None, **kwargs):
            captured_sizes.append(params.get("pageSize"))
            resp = MagicMock(spec=httpx.Response)
            resp.status_code = 200
            resp.json.return_value = {"studies": []}
            resp.raise_for_status = MagicMock()
            return resp

        with patch("httpx.Client.get", side_effect=mock_get):
            fetch_trials(["cancer"], page_size=-10)
            fetch_trials(["cancer"], page_size=0)
            fetch_trials(["cancer"], page_size=250)
            fetch_trials(["cancer"], page_size=5000)

        assert captured_sizes == [1, 1, 250, 1000]

    def test_status_filter_options(self):
        captured_statuses = []

        def mock_get(url, params=None, **kwargs):
            captured_statuses.append(params.get("filter.overallStatus"))
            resp = MagicMock(spec=httpx.Response)
            resp.status_code = 200
            resp.json.return_value = {"studies": []}
            resp.raise_for_status = MagicMock()
            return resp

        with patch("httpx.Client.get", side_effect=mock_get):
            fetch_trials(["cancer"], status="RECRUITING")
            fetch_trials(["cancer"], status="COMPLETED")
            fetch_trials(["cancer"], status="ALL")
            fetch_trials(["cancer"], status=None)

        assert captured_statuses == ["RECRUITING", "COMPLETED", None, None]

    def test_headers_and_fields_projection(self):
        captured_headers = {}
        captured_fields = ""

        def mock_get(url, params=None, headers=None, **kwargs):
            nonlocal captured_fields
            captured_fields = params.get("fields", "")
            resp = MagicMock(spec=httpx.Response)
            resp.status_code = 200
            resp.json.return_value = {"studies": []}
            resp.raise_for_status = MagicMock()
            return resp

        with patch("httpx.Client") as mock_client_cls:
            mock_instance = MagicMock()
            mock_instance.get.side_effect = mock_get
            mock_instance.__enter__.return_value = mock_instance
            mock_instance.__exit__.return_value = None
            mock_client_cls.return_value = mock_instance

            fetch_trials(["diabetes"])

            call_kwargs = mock_client_cls.call_args[1]
            captured_headers = call_kwargs.get("headers", {})
            timeout = call_kwargs.get("timeout")

        assert captured_headers.get("User-Agent") == USER_AGENT
        assert timeout == DEFAULT_TIMEOUT
        assert "NCTId" in captured_fields
        assert "BriefTitle" in captured_fields
        assert "EligibilityCriteria" in captured_fields
        assert "OverallStatus" in captured_fields

    def test_http_400_bad_request_raises_error(self):
        def mock_get(url, params=None, **kwargs):
            resp = MagicMock(spec=httpx.Response)
            resp.status_code = 400
            resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                "400 Bad Request",
                request=MagicMock(spec=httpx.Request),
                response=resp,
            )
            return resp

        with patch("httpx.Client.get", side_effect=mock_get):
            with pytest.raises(httpx.HTTPStatusError) as exc_info:
                fetch_trials(["cancer"])
            assert exc_info.value.response.status_code == 400

    def test_http_404_not_found_raises_error(self):
        def mock_get(url, params=None, **kwargs):
            resp = MagicMock(spec=httpx.Response)
            resp.status_code = 404
            resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                "404 Not Found",
                request=MagicMock(spec=httpx.Request),
                response=resp,
            )
            return resp

        with patch("httpx.Client.get", side_effect=mock_get):
            with pytest.raises(httpx.HTTPStatusError) as exc_info:
                fetch_trials(["cancer"])
            assert exc_info.value.response.status_code == 404

    def test_http_500_internal_error_raises_error(self):
        def mock_get(url, params=None, **kwargs):
            resp = MagicMock(spec=httpx.Response)
            resp.status_code = 500
            resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                "500 Internal Server Error",
                request=MagicMock(spec=httpx.Request),
                response=resp,
            )
            return resp

        with patch("httpx.Client.get", side_effect=mock_get):
            with pytest.raises(httpx.HTTPStatusError) as exc_info:
                fetch_trials(["cancer"])
            assert exc_info.value.response.status_code == 500

    def test_http_429_rate_limit_retries_and_succeeds(self, sample_studies):
        call_count = 0

        def mock_get(url, params=None, **kwargs):
            nonlocal call_count
            call_count += 1
            resp = MagicMock(spec=httpx.Response)
            if call_count == 1:
                resp.status_code = 429
                resp.headers = {"Retry-After": "0"}
            else:
                resp.status_code = 200
                resp.json.return_value = {"studies": sample_studies}
                resp.raise_for_status = MagicMock()
            return resp

        with patch("httpx.Client.get", side_effect=mock_get), patch("time.sleep") as mock_sleep:
            results = fetch_trials(["cancer"], backoff_factor=0.01)
            assert len(results) == len(sample_studies)
            assert call_count == 2
            mock_sleep.assert_called_once()

    def test_http_503_service_unavailable_retries_and_succeeds(self, sample_studies):
        call_count = 0

        def mock_get(url, params=None, **kwargs):
            nonlocal call_count
            call_count += 1
            resp = MagicMock(spec=httpx.Response)
            if call_count < 3:
                resp.status_code = 503
                resp.headers = {}
            else:
                resp.status_code = 200
                resp.json.return_value = {"studies": sample_studies}
                resp.raise_for_status = MagicMock()
            return resp

        with patch("httpx.Client.get", side_effect=mock_get), patch("time.sleep") as mock_sleep:
            results = fetch_trials(["heart disease"], backoff_factor=0.01)
            assert len(results) == len(sample_studies)
            assert call_count == 3
            assert mock_sleep.call_count == 2

    def test_http_429_exhausted_retries_raises_status_error(self):
        def mock_get(url, params=None, **kwargs):
            resp = MagicMock(spec=httpx.Response)
            resp.status_code = 429
            resp.headers = {}
            resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                "429 Too Many Requests",
                request=MagicMock(spec=httpx.Request),
                response=resp,
            )
            return resp

        with patch("httpx.Client.get", side_effect=mock_get), patch("time.sleep"):
            with pytest.raises(httpx.HTTPStatusError) as exc_info:
                fetch_trials(["cancer"], max_retries=2, backoff_factor=0.01)
            assert exc_info.value.response.status_code == 429

    def test_network_connection_error_retries_and_raises(self):
        call_count = 0

        def mock_get(url, params=None, **kwargs):
            nonlocal call_count
            call_count += 1
            raise httpx.ConnectError("Connection refused")

        with patch("httpx.Client.get", side_effect=mock_get), patch("time.sleep"):
            with pytest.raises(httpx.ConnectError):
                fetch_trials(["diabetes"], max_retries=2, backoff_factor=0.01)
            assert call_count == 3


# ============================================================================
# 4. Standard Category Pre-loader Unit Tests (preload_standard_trials)
# ============================================================================

class TestPreloadStandardTrials:
    """Test preload_standard_trials querying, deduplication, and resilience."""

    def test_preload_default_categories(self):
        queried_categories = []

        def mock_fetch(conditions, page_size=200, status="RECRUITING"):
            queried_categories.extend(conditions)
            return [make_mock_study(nct_id=f"NCT_{conditions[0]}")]

        with patch("core.trial_fetcher.fetch_trials", side_effect=mock_fetch):
            results = preload_standard_trials()

        assert len(results) == 4
        assert set(queried_categories) == {
            "cancer",
            "diabetes",
            "heart disease",
            "neurological",
        }

    def test_preload_custom_categories_and_proportional_page_size(self):
        queried_calls = []

        def mock_fetch(conditions, page_size=200, status="RECRUITING"):
            queried_calls.append({"cat": conditions[0], "page_size": page_size})
            return []

        with patch("core.trial_fetcher.fetch_trials", side_effect=mock_fetch):
            preload_standard_trials(categories=["rare disease", "pediatrics"], page_size=100)

        assert len(queried_calls) == 2
        assert queried_calls[0] == {"cat": "rare disease", "page_size": 50}
        assert queried_calls[1] == {"cat": "pediatrics", "page_size": 50}

    def test_preload_empty_categories_returns_empty(self):
        results = preload_standard_trials(categories=[])
        assert results == []

    def test_preload_deduplication_across_overlapping_categories(self, sample_studies):
        shared_study = sample_studies[0]  # NCT00000001
        cancer_study = sample_studies[1]
        diabetes_study = sample_studies[2]

        def mock_fetch(conditions, page_size=200, status="RECRUITING"):
            cat = conditions[0]
            if cat == "cancer":
                return [shared_study, cancer_study]
            elif cat == "diabetes":
                return [shared_study, diabetes_study]
            return []

        with patch("core.trial_fetcher.fetch_trials", side_effect=mock_fetch):
            results = preload_standard_trials(categories=["cancer", "diabetes"])

        assert len(results) == 3
        ids = [s["protocolSection"]["identificationModule"]["nctId"] for s in results]
        assert ids.count("NCT00000001") == 1

    def test_preload_handles_single_category_failure_gracefully(self, sample_studies):
        def mock_fetch(conditions, page_size=200, status="RECRUITING"):
            cat = conditions[0]
            if cat == "cancer":
                raise httpx.ConnectError("Network timeout on cancer query")
            return [sample_studies[1]]

        with patch("core.trial_fetcher.fetch_trials", side_effect=mock_fetch):
            results = preload_standard_trials(categories=["cancer", "diabetes"])

        assert len(results) == 1
        assert results[0]["protocolSection"]["identificationModule"]["nctId"] == "NCT00000002"
