import os
import uuid
import requests


BASE_URL = os.getenv("DEPLOY_BASE_URL", "http://localhost:8000").rstrip("/")


def test_health_or_root():
    """Проверяем доступность приложения по / (fallback вместо /health)."""
    response = requests.get(f"{BASE_URL}/", timeout=15)
    assert response.status_code == 200


def test_post_valid_data():
    """Проверяем POST с валидными данными на endpoint регистрации."""
    payload = {
        "name": "deploy-test-user",
        "email": f"deploy-test-{uuid.uuid4().hex[:10]}@example.com",
        "password": "StrongPass123!",
    }
    response = requests.post(f"{BASE_URL}/register", json=payload, timeout=15)
    assert response.status_code in (200, 201), response.text


def test_post_invalid_data_returns_400():
    """Проверяем POST с невалидными данными."""
    invalid_payload = {"email": "", "password": ""}
    response = requests.post(f"{BASE_URL}/login", json=invalid_payload, timeout=15)
    assert response.status_code == 400, response.text
