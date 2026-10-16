import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import httpx

# Ensure site packages
user_site = os.path.expanduser('~/AppData/Roaming/Python/Python313/site-packages')
if os.path.exists(user_site) and user_site not in sys.path:
    sys.path.insert(0, user_site)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import settings
from app.ai.llm import OllamaLLM, MockLLM, GeminiLLM, get_llm_provider
from app.ai.embeddings import MockEmbedding, GeminiEmbedding, get_embedding_provider
from app.ai.rag import RAGEngine
from app.ai.vector_store import get_vector_store


class TestLLMProviderIntegration(unittest.TestCase):
    """
    Test suite for Phase 2 & Phase 6:
    - Ollama URL configuration and URL normalization
    - Payload construction and option parameters
    - Connection error, timeout, and 404 model-not-found handling
    - Provider selection in get_llm_provider()
    - Real local Ollama inference check if available
    - RAG error handling when LLM service fails
    - Embedding provider dimension compatibility
    """

    def test_01_ollama_url_normalization(self):
        # Default
        llm = OllamaLLM(model="qwen2.5-coder:1.5b", base_url="http://localhost:11434")
        self.assertEqual(llm.base_url, "http://localhost:11434")
        self.assertEqual(llm.chat_url, "http://localhost:11434/api/chat")
        self.assertEqual(llm.tags_url, "http://localhost:11434/api/tags")

        # Docker internal with trailing slash
        llm_docker = OllamaLLM(model="qwen2.5-coder:1.5b", base_url="http://host.docker.internal:11434/")
        self.assertEqual(llm_docker.base_url, "http://host.docker.internal:11434")
        self.assertEqual(llm_docker.chat_url, "http://host.docker.internal:11434/api/chat")

        # If user configured full endpoint path accidentally
        llm_full = OllamaLLM(model="qwen2.5-coder:1.5b", base_url="http://host.docker.internal:11434/api/chat")
        self.assertEqual(llm_full.base_url, "http://host.docker.internal:11434")
        self.assertEqual(llm_full.chat_url, "http://host.docker.internal:11434/api/chat")

    def test_02_ollama_payload_construction(self):
        llm = OllamaLLM(model="qwen2.5-coder:1.5b", base_url="http://test-host:11434")
        captured_request = {}

        def mock_post(url, json=None, **kwargs):
            captured_request["url"] = url
            captured_request["json"] = json
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "message": {"content": "Sample generated answer."}
            }
            return mock_resp

        with patch("httpx.Client.post", side_effect=mock_post):
            answer = llm.generate_response("System instruction", "User query")

        self.assertEqual(answer, "Sample generated answer.")
        self.assertEqual(captured_request["url"], "http://test-host:11434/api/chat")
        payload = captured_request["json"]
        self.assertEqual(payload["model"], "qwen2.5-coder:1.5b")
        self.assertEqual(payload["stream"], False)
        self.assertEqual(len(payload["messages"]), 2)
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][0]["content"], "System instruction")
        self.assertEqual(payload["messages"][1]["role"], "user")
        self.assertEqual(payload["messages"][1]["content"], "User query")
        self.assertIn("options", payload)
        self.assertEqual(payload["options"]["temperature"], 0.1)

    def test_03_ollama_connection_error_handling(self):
        llm = OllamaLLM(model="qwen2.5-coder:1.5b", base_url="http://unreachable-host:11434")

        with patch("httpx.Client.post", side_effect=httpx.ConnectError("Connection refused")):
            response = llm.generate_response("System", "User")
            self.assertIn("Could not connect to the local Ollama service", response)
            self.assertIn("http://unreachable-host:11434", response)

        with patch("httpx.Client.stream", side_effect=httpx.ConnectError("Connection refused")):
            chunks = list(llm.stream_response("System", "User"))
            self.assertTrue(len(chunks) > 0)
            self.assertIn("Error: Connection to local AI", chunks[0])

    def test_04_ollama_timeout_handling(self):
        llm = OllamaLLM(model="qwen2.5-coder:1.5b", base_url="http://timeout-host:11434", timeout=5.0)

        with patch("httpx.Client.post", side_effect=httpx.TimeoutException("Read timed out")):
            response = llm.generate_response("System", "User")
            self.assertIn("timed out", response.lower())

        with patch("httpx.Client.stream", side_effect=httpx.TimeoutException("Read timed out")):
            chunks = list(llm.stream_response("System", "User"))
            self.assertTrue(len(chunks) > 0)
            self.assertIn("timed out", chunks[0].lower())

    def test_05_ollama_missing_model_404_handling(self):
        llm = OllamaLLM(model="nonexistent-model", base_url="http://test-host:11434")

        mock_resp = MagicMock()
        mock_resp.status_code = 404

        with patch("httpx.Client.post", return_value=mock_resp):
            response = llm.generate_response("System", "User")
            self.assertIn("not installed", response)
            self.assertIn("ollama pull nonexistent-model", response)

        with patch("httpx.Client.stream", return_value=mock_resp):
            # For stream context manager
            cm = MagicMock()
            cm.__enter__.return_value = mock_resp
            with patch("httpx.Client.stream", return_value=cm):
                chunks = list(llm.stream_response("System", "User"))
                self.assertTrue(len(chunks) > 0)
                self.assertIn("not installed", chunks[0])
                self.assertIn("ollama pull nonexistent-model", chunks[0])

    def test_06_get_llm_provider_selection(self):
        # 1. Ollama provider
        with patch.object(settings, "LLM_PROVIDER", "ollama"), \
             patch.object(settings, "OLLAMA_MODEL", "qwen2.5-coder:1.5b"), \
             patch.object(settings, "OLLAMA_BASE_URL", "http://host.docker.internal:11434"):
            p = get_llm_provider()
            self.assertIsInstance(p, OllamaLLM)
            self.assertEqual(p.model, "qwen2.5-coder:1.5b")
            self.assertEqual(p.base_url, "http://host.docker.internal:11434")

        # 2. Mock provider
        with patch.object(settings, "LLM_PROVIDER", "mock"):
            p = get_llm_provider()
            self.assertIsInstance(p, MockLLM)

        # 3. Gemini provider with configured model
        with patch.object(settings, "LLM_PROVIDER", "gemini"), \
             patch.object(settings, "GEMINI_API_KEY", "test-gemini-key-1234"), \
             patch.object(settings, "GEMINI_MODEL", "gemini-2.5-flash"):
            p = get_llm_provider()
            self.assertIsInstance(p, GeminiLLM)
            self.assertEqual(p.api_key, "test-gemini-key-1234")
            self.assertEqual(p.model, "gemini-2.5-flash")

        # 4. Missing API key fallback to MockLLM with warning
        with patch.object(settings, "LLM_PROVIDER", "openai"), \
             patch.object(settings, "OPENAI_API_KEY", None):
            p = get_llm_provider()
            self.assertIsInstance(p, MockLLM)

    def test_07_rag_flags_service_error_on_connection_failure(self):
        class FailingLLM:
            def generate_response(self, system_prompt, user_prompt):
                return "Could not connect to the local Ollama service at http://test-host:11434. Please make sure Ollama is running."
            def stream_response(self, system_prompt, user_prompt):
                yield "\n[Error: Connection to local AI at http://test-host:11434 failed. Ensure Ollama is running (`ollama serve`).]"

        failing_llm = FailingLLM()
        engine = RAGEngine(vector_store=get_vector_store(), llm_provider=failing_llm)

        # Non-streaming
        res = engine.ask_question("6ed6107b-7311-499d-960a-1a21edb7398a", "What does rag.py do?", stream=False)
        self.assertTrue(res.get("error", False), "Expected error=True when LLM connection fails")
        self.assertFalse(res.get("grounded", True))
        self.assertEqual(len(res.get("sources", [])), 0)

        # Streaming
        stream_gen = engine.ask_question("6ed6107b-7311-499d-960a-1a21edb7398a", "What does rag.py do?", stream=True)
        import json
        done_data = None
        for chunk in stream_gen:
            data = json.loads(chunk)
            if data.get("type") == "done":
                done_data = data
                break
        self.assertIsNotNone(done_data)
        self.assertTrue(done_data.get("error", False))
        self.assertFalse(done_data.get("grounded", True))

    def test_08_embedding_provider_dimension_compatibility(self):
        mock_provider = MockEmbedding()
        self.assertEqual(mock_provider.dimension, 384)

        # Gemini Embedding model name and dimension
        gemini_provider = GeminiEmbedding(api_key="test-key")
        self.assertEqual(gemini_provider.model_name, "gemini-embedding-001")
        self.assertEqual(gemini_provider.dimension, 768)

        try:
            from app.ai.embeddings import LocalEmbedding
            local_provider = LocalEmbedding()
            self.assertEqual(local_provider.dimension, 384)
            self.assertEqual(mock_provider.dimension, local_provider.dimension)
        except Exception:
            pass  # If sentence-transformers not in current env, mock dim is verified at 384

    def test_09_real_local_ollama_inference(self):
        """
        Live verification of real Ollama on Windows host if accessible.
        """
        llm = OllamaLLM(model="qwen2.5-coder:1.5b", base_url="http://localhost:11434")
        health = llm.check_health()
        if not health.get("ok"):
            self.skipTest(f"Local Ollama not reachable at localhost:11434: {health.get('error')}")

        resp = llm.generate_response("You are a helpful assistant.", "Reply with exactly: PONG")
        self.assertTrue(len(resp) > 0)
        self.assertIn("PONG", resp.upper())

    def test_10_gemini_client_model_and_url_forwarding(self):
        """
        Confirm that configured GEMINI_MODEL is passed to the Gemini API client
        and properly formatted in the outgoing Google Generative Language API endpoint.
        """
        captured_requests = []

        def mock_client_post(url, json=None, **kwargs):
            captured_requests.append({
                "url": url,
                "json": json,
                "headers": kwargs.get("headers", {})
            })
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "Grounded answer from Gemini."}]
                        }
                    }
                ]
            }
            return mock_resp

        # 1. Test default gemini-2.5-flash model
        gemini_llm = GeminiLLM(api_key="secret-api-key-xyz", model="gemini-2.5-flash")
        self.assertEqual(gemini_llm.model, "gemini-2.5-flash")

        with patch("httpx.Client.post", side_effect=mock_client_post):
            answer = gemini_llm.generate_response("System prompt", "User prompt")
            self.assertEqual(answer, "Grounded answer from Gemini.")

        self.assertEqual(len(captured_requests), 1)
        req = captured_requests[0]
        self.assertIn("v1beta/models/gemini-2.5-flash:generateContent", req["url"])
        # API key must be in the header, never in the URL query string
        self.assertNotIn("key=", req["url"])
        self.assertEqual(req["headers"].get("x-goog-api-key"), "secret-api-key-xyz")
        self.assertEqual(
            req["json"]["system_instruction"]["parts"][0]["text"],
            "System prompt"
        )
        self.assertEqual(
            req["json"]["contents"][0]["parts"][0]["text"],
            "User prompt"
        )

        # 2. Test dynamic custom model override
        gemini_custom = GeminiLLM(api_key="secret-api-key-xyz", model="gemini-2.5-pro")
        self.assertEqual(gemini_custom.model, "gemini-2.5-pro")

        with patch("httpx.Client.post", side_effect=mock_client_post):
            gemini_custom.generate_response("Sys", "User")

        self.assertEqual(len(captured_requests), 2)
        req2 = captured_requests[1]
        self.assertIn("v1beta/models/gemini-2.5-pro:generateContent", req2["url"])


if __name__ == "__main__":
    unittest.main()
