import pytest
from fastapi.testclient import TestClient
from app.main import create_app


@pytest.fixture()
def db_url(tmp_path):
    return f"sqlite:///{tmp_path}/t.db"


@pytest.fixture()
def client(db_url):
    return TestClient(create_app(db_url, frontend_dir="/nonexistent"))
