from fastapi.testclient import TestClient

from app import app


def main() -> None:
    client = TestClient(app)
    response = client.get("/health")
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "ok":
        raise RuntimeError(f"Unexpected payload: {payload}")
    print("Smoke test passed: /health returned status=ok")


if __name__ == "__main__":
    main()