"""API integration: health, readiness, model listing, and the 501 contracts.

Runs the real application against a temporary artifact root. No network, no
database, no model weights.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from omnirank.api.app import create_app
from omnirank.api.middleware import REQUEST_ID_HEADER
from omnirank.artifacts.metadata import ArtifactMetadata, SupportedDevice

pytestmark = pytest.mark.integration


@pytest.fixture
def client(config, artifact_root, monkeypatch, tmp_path):
    """A client whose app resolves paths inside tmp_path."""
    monkeypatch.chdir(tmp_path)
    with TestClient(create_app(config), raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def ready_client(config, artifact_root, sample_metadata, monkeypatch, tmp_path):
    """A client with one compatible artifact registered, so /ready passes."""
    monkeypatch.chdir(tmp_path)
    from omnirank.artifacts.registry import ArtifactRegistry

    ArtifactRegistry(artifact_root / "metadata", artifact_root=artifact_root).register(
        sample_metadata
    )
    with TestClient(create_app(config), raise_server_exceptions=False) as test_client:
        yield test_client


class TestHealth:
    def test_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_reports_real_process_state(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["service"] == "omnirank"
        assert body["environment"] == "local"
        assert body["uptime_seconds"] >= 0

    def test_does_not_depend_on_artifacts(self, client):
        """Liveness must stay green when the service is not ready."""
        assert client.get("/ready").status_code == 503
        assert client.get("/health").status_code == 200


class TestReadiness:
    def test_503_with_no_artifacts(self, client):
        response = client.get("/ready")
        assert response.status_code == 503
        assert response.json()["ready"] is False

    def test_names_the_unready_dependency(self, client):
        dependencies = {d["name"]: d for d in client.get("/ready").json()["dependencies"]}
        assert dependencies["artifact_registry"]["ready"] is False
        assert "no artifacts registered" in dependencies["artifact_registry"]["detail"]

    def test_configuration_and_device_are_ready(self, client):
        dependencies = {d["name"]: d for d in client.get("/ready").json()["dependencies"]}
        assert dependencies["configuration"]["ready"] is True
        assert dependencies["device"]["ready"] is True

    def test_reports_no_unchecked_dependency_as_ready(self, client):
        """Postgres/Redis have no client in Phase 1, so they must not be claimed."""
        names = {d["name"] for d in client.get("/ready").json()["dependencies"]}
        assert "postgres" not in names
        assert "redis" not in names

    def test_200_once_a_compatible_artifact_exists(self, ready_client):
        response = ready_client.get("/ready")
        assert response.status_code == 200
        assert response.json()["ready"] is True

    def test_incompatible_artifact_does_not_make_the_service_ready(
        self, config, artifact_root, sample_metadata, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        from omnirank.artifacts.registry import ArtifactRegistry

        payload = sample_metadata.model_dump()
        payload["supported_device"] = SupportedDevice.CUDA
        ArtifactRegistry(artifact_root / "metadata", artifact_root=artifact_root).register(
            ArtifactMetadata.model_validate(payload)
        )
        with TestClient(create_app(config), raise_server_exceptions=False) as client:
            body = client.get("/ready").json()
            assert body["ready"] is False
            detail = next(d for d in body["dependencies"] if d["name"] == "artifact_registry")[
                "detail"
            ]
            assert "none compatible" in detail


class TestModels:
    def test_empty_registry_returns_an_empty_list(self, client):
        body = client.get("/v1/models").json()
        assert body == {
            "models": [],
            "count": 0,
            "device": body["device"],
            "serving_ready": False,
        }

    def test_lists_a_registered_artifact(self, ready_client):
        body = ready_client.get("/v1/models").json()
        assert body["count"] == 1
        assert body["models"][0]["model_name"] == "popularity"
        assert body["models"][0]["compatible"] is True

    def test_reports_only_recorded_metrics(self, ready_client):
        """Metrics come from the manifest; none are invented."""
        assert ready_client.get("/v1/models").json()["models"][0]["metrics"] == {"recall@20": 0.11}

    def test_incompatible_artifact_is_listed_with_a_reason(
        self, config, artifact_root, sample_metadata, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        from omnirank.artifacts.registry import ArtifactRegistry

        payload = sample_metadata.model_dump()
        payload["supported_device"] = SupportedDevice.CUDA
        ArtifactRegistry(artifact_root / "metadata", artifact_root=artifact_root).register(
            ArtifactMetadata.model_validate(payload)
        )
        with TestClient(create_app(config), raise_server_exceptions=False) as client:
            summary = client.get("/v1/models").json()["models"][0]
            assert summary["compatible"] is False
            assert "cuda" in summary["incompatibility_reason"]


# Every endpoint whose contract is declared but not implemented.
UNIMPLEMENTED_CASES = [
    ("GET", "/v1/recommendations/users/u1", None),
    ("GET", "/v1/recommendations/similar/i1", None),
    ("POST", "/v1/recommendations/session", {"session_id": "s1", "item_ids": ["i1"]}),
    (
        "POST",
        "/v1/interactions",
        {"events": [{"user_id": "u1", "item_id": "i1", "event_type": "click"}]},
    ),
    ("GET", "/v1/items/i1", None),
    ("POST", "/v1/admin/reload-artifacts", {}),
]


class TestUnimplementedContracts:
    """Endpoints that exist as contracts must return 501, never fabricated data."""

    @pytest.mark.parametrize(("method", "path", "body"), UNIMPLEMENTED_CASES)
    def test_returns_501(self, client, method, path, body):
        response = client.request(method, path, json=body)
        assert response.status_code == 501

    @pytest.mark.parametrize(("method", "path", "body"), UNIMPLEMENTED_CASES)
    def test_names_the_delivering_phase(self, client, method, path, body):
        error = client.request(method, path, json=body).json()["error"]
        assert error["code"] == "not_implemented_yet"
        assert error["context"]["planned_phase"] >= 2

    @pytest.mark.parametrize(("method", "path", "body"), UNIMPLEMENTED_CASES)
    def test_never_returns_a_recommendation_payload(self, client, method, path, body):
        assert "recommendations" not in client.request(method, path, json=body).json()

    def test_501_is_reached_after_schema_validation_not_before(self, client):
        """Malformed input still fails as 422; the contract is enforced first."""
        response = client.post("/v1/interactions", json={"events": []})
        assert response.status_code == 422


class TestErrorHandling:
    def test_unknown_path_is_404(self, client):
        assert client.get("/v1/nonexistent").status_code == 404

    def test_invalid_query_parameter_is_422(self, client):
        assert client.get("/v1/recommendations/users/u1?k=0").status_code == 422

    def test_k_above_the_cap_is_422(self, client):
        assert client.get("/v1/recommendations/users/u1?k=99999").status_code == 422

    def test_error_bodies_share_one_shape(self, client):
        error = client.get("/v1/items/i1").json()["error"]
        assert set(error) == {"code", "message", "context", "request_id"}


class TestRequestCorrelation:
    def test_every_response_carries_a_request_id(self, client):
        assert client.get("/health").headers[REQUEST_ID_HEADER]

    def test_a_caller_supplied_id_is_honoured(self, client):
        response = client.get("/health", headers={REQUEST_ID_HEADER: "trace-abc"})
        assert response.headers[REQUEST_ID_HEADER] == "trace-abc"

    def test_an_over_long_id_is_truncated(self, client):
        response = client.get("/health", headers={REQUEST_ID_HEADER: "x" * 500})
        assert len(response.headers[REQUEST_ID_HEADER]) == 64

    def test_ids_differ_between_requests(self, client):
        first = client.get("/health").headers[REQUEST_ID_HEADER]
        second = client.get("/health").headers[REQUEST_ID_HEADER]
        assert first != second

    def test_error_bodies_include_the_request_id(self, client):
        response = client.get("/v1/items/i1", headers={REQUEST_ID_HEADER: "trace-xyz"})
        assert response.json()["error"]["request_id"] == "trace-xyz"

    def test_response_time_header_is_present(self, client):
        assert float(client.get("/health").headers["X-Response-Time-ms"]) >= 0


class TestOpenApi:
    def test_document_builds(self, client):
        assert client.get("/openapi.json").status_code == 200

    def test_every_planned_endpoint_is_documented(self, client):
        paths = client.get("/openapi.json").json()["paths"]
        assert set(paths) == {
            "/health",
            "/ready",
            "/v1/models",
            "/v1/recommendations/users/{user_id}",
            "/v1/recommendations/similar/{item_id}",
            "/v1/recommendations/session",
            "/v1/interactions",
            "/v1/items/{item_id}",
            "/v1/admin/reload-artifacts",
        }

    def test_unimplemented_endpoints_are_marked_in_their_description(self, client):
        paths = client.get("/openapi.json").json()["paths"]
        description = paths["/v1/recommendations/users/{user_id}"]["get"]["description"]
        assert "Not implemented in Phase 1" in description

    def test_recommendation_response_schema_matches_the_agreed_shape(self, client):
        schema = client.get("/openapi.json").json()["components"]["schemas"][
            "RecommendationResponse"
        ]
        assert {
            "user_id",
            "model_version",
            "recommendations",
            "fallback_used",
            "latency_ms",
        } <= set(schema["properties"])

    def test_recommendation_item_carries_sources_and_reason(self, client):
        schema = client.get("/openapi.json").json()["components"]["schemas"]["RecommendationItem"]
        assert {"item_id", "rank", "score", "sources", "reason"} <= set(schema["properties"])

    def test_docs_page_is_served(self, client):
        assert client.get("/docs").status_code == 200
