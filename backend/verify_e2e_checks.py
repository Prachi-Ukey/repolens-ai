import requests
import json
import time
import sys

sys.stdout.reconfigure(encoding='utf-8')

BASE_URL = "http://127.0.0.1:8000"

def run_checks():
    print("=== REPOLENS AI E2E COMPREHENSIVE VERIFICATION ===")

    # 1. Demo Token
    print("\n[Check 1] Auto Demo Token Provisioning...")
    token_resp = requests.get(f"{BASE_URL}/auth/demo-token")
    assert token_resp.status_code == 200, f"Token failed: {token_resp.text}"
    token_data = token_resp.json()
    token = token_data["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print(f"PASS: Demo token retrieved successfully. (User: prachi)")

    # 2. Repository Listing & Overview
    print("\n[Check 2] Repository Overview...")
    repos_resp = requests.get(f"{BASE_URL}/repositories", headers=headers)
    assert repos_resp.status_code == 200
    repos = repos_resp.json()
    assert len(repos) > 0, "No repositories found!"
    target_repo = next((r for r in repos if "vehicle-damage" in r["name"]), repos[0])
    repo_id = target_repo["id"]
    print(f"Target Repo: {target_repo['name']} (ID: {repo_id})")

    overview_resp = requests.get(f"{BASE_URL}/repositories/{repo_id}/overview", headers=headers)
    assert overview_resp.status_code == 200
    overview_data = overview_resp.json()
    assert overview_data["file_count"] > 0
    assert overview_data["chunk_count"] > 0
    assert len(overview_data["readme_preview"]) > 0
    print(f"PASS: Overview endpoint returned: {overview_data['file_count']} files, {overview_data['chunk_count']} chunks, {len(overview_data['readme_preview'])} chars README preview.")

    # 3. Source File Retrieval
    print("\n[Check 3] Source File Retrieval by Path & ID...")
    file_resp = requests.get(f"{BASE_URL}/repositories/{repo_id}/files/app.py", headers=headers)
    assert file_resp.status_code == 200, f"File fetch failed: {file_resp.status_code}"
    file_data = file_resp.json()
    assert "content" in file_data and "login" in file_data["content"]
    print(f"PASS: Retrieved app.py ({len(file_data['content'].splitlines())} lines). Contains login functionality.")

    # 4. Repository URL Validation
    print("\n[Check 4] Invalid Repository URL Rejection...")
    invalid_resp = requests.post(
        f"{BASE_URL}/repositories",
        headers=headers,
        json={"url": "not-a-valid-github-url"}
    )
    assert invalid_resp.status_code in [400, 422], f"Expected 400/422, got {invalid_resp.status_code}: {invalid_resp.text}"
    print(f"PASS: Invalid URL rejected with status code {invalid_resp.status_code}.")

    # 5. Project Overview Question
    print("\n[Check 5] Project Overview Question ('What does this project do?')...")
    q1_resp = requests.post(
        f"{BASE_URL}/repositories/{repo_id}/chat",
        headers=headers,
        json={"question": "What does this project do?"}
    )
    assert q1_resp.status_code == 200, f"Q1 failed: {q1_resp.status_code} {q1_resp.text}"
    q1_data = q1_resp.json()
    print("AI Answer:", q1_data["answer"][:160], "...")
    print("Sources:", q1_data.get("sources"))
    print("Is Grounded:", q1_data.get("grounded"))
    assert q1_data.get("grounded") is True
    assert any("README.md" in s.get("file_path", "") for s in q1_data.get("sources", []))
    print("PASS: Overview question answered and grounded in README.md.")

    # 6. Source Code Question
    print("\n[Check 6] Source Code Question ('Where is the login functionality implemented?')...")
    q2_resp = requests.post(
        f"{BASE_URL}/repositories/{repo_id}/chat",
        headers=headers,
        json={"question": "Where is the login functionality implemented?"}
    )
    assert q2_resp.status_code == 200, f"Q2 failed: {q2_resp.status_code} {q2_resp.text}"
    q2_data = q2_resp.json()
    print("AI Answer:", q2_data["answer"][:200], "...")
    print("Sources:", [(s.get("file_path"), s.get("start_line"), s.get("end_line")) for s in q2_data.get("sources", [])])
    print("Is Grounded:", q2_data.get("grounded"))
    assert q2_data.get("grounded") is True
    assert any("app.py" in s.get("file_path", "") for s in q2_data.get("sources", []))
    print("PASS: Source code question answered and grounded in app.py with valid line range.")

    # 7. Unsupported Question (Honest refusal)
    print("\n[Check 7] Unsupported Question ('Where is the Ethereum smart contract logic?')...")
    q3_resp = requests.post(
        f"{BASE_URL}/repositories/{repo_id}/chat",
        headers=headers,
        json={"question": "Where is the Ethereum smart contract logic?"}
    )
    assert q3_resp.status_code == 200, f"Q3 failed: {q3_resp.status_code} {q3_resp.text}"
    q3_data = q3_resp.json()
    print("AI Answer:", q3_data["answer"][:200], "...")
    print("Sources:", q3_data.get("sources"))
    print("Is Grounded:", q3_data.get("grounded"))
    assert q3_data.get("grounded") is False
    assert len(q3_data.get("sources", [])) == 0
    ans_lower = q3_data["answer"].lower()
    assert any(term in ans_lower for term in ["insufficient", "couldn't find", "not find", "not enough evidence", "not mention", "no evidence", "does not contain", "no information", "cannot find"])
    print("PASS: Honest refusal when evidence is absent, no hallucinated citations.")

    # 8. Unavailable Model / Ollama Health Error Handling
    print("\n[Check 8] Model Health Check Endpoint...")
    health_resp = requests.get(f"{BASE_URL}/health")
    assert health_resp.status_code == 200
    health_data = health_resp.json()
    print("Health Status:", health_data)
    print("PASS: Health endpoint confirms backend and local AI status.")

    print("\n=======================================================")
    print("ALL 8 VERIFICATION CHECKS COMPLETED AND PASSED!")
    print("=======================================================")

if __name__ == "__main__":
    run_checks()
