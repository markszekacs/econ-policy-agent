"""Basic API tests."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from api.main import app
from database.connection import init_db, engine
from database.models import Base

@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_list_runs_empty():
    response = client.get("/runs")
    assert response.status_code == 200
    assert response.json() == []


def test_get_run_not_found():
    response = client.get("/runs/nonexistent-id")
    assert response.status_code == 404


def test_analyze_empty_question():
    response = client.post(
        "/analyze",
        json={"question": "   "}
    )
    assert response.status_code == 400
