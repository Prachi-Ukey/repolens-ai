from abc import ABC, abstractmethod
import logging
import re
from typing import List, Dict, Any, Optional

import httpx

from app.config import settings


logger = logging.getLogger("repolens.ai.llm")


class LLMProvider(ABC):

    @abstractmethod
    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> str:
        pass


class MockLLM(LLMProvider):
    """
    Deterministic fallback LLM for offline/demo testing.
    """

    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> str:

        chunk_matches = re.findall(
            r"--- CHUNK \d+ \| File: (.*?) \| Lines: "
            r"(\d+)-(\d+) \| Symbol: (.*?) ---\n"
            r"(.*?)(?=\n--- CHUNK|\n</repository_code_context>|$)",
            user_prompt,
            re.DOTALL
        )

        if not chunk_matches:
            return (
                "I couldn't find enough evidence in the repository "
                "to answer this confidently."
            )

        first = chunk_matches[0]

        file_path = first[0]
        start_line = first[1]
        end_line = first[2]

        citation = (
            f"[{file_path}:{start_line}-{end_line}]"
        )

        return (
            "The relevant implementation is available in "
            f"`{file_path}` {citation}."
        )


class OllamaLLM(LLMProvider):
    """
    Local LLM provider using Ollama.
    """

    def __init__(
        self,
        model: str = "qwen2.5-coder:7b"
    ):
        self.model = model

        self.url = (
            "http://localhost:11434/api/chat"
        )

    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> str:

        payload = {
            "model": self.model,

            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],

            "stream": False,

            "options": {
                "temperature": 0.1,

                # Smaller response reduces repetition
                # and hallucination for the local model.
                "num_predict": 256,

                "num_ctx": 4096,

                "top_p": 0.9
            }
        }

        logger.info(
            "Sending RAG prompt to local Ollama "
            "model '%s'...",
            self.model
        )

        try:

            with httpx.Client(
                timeout=120.0
            ) as client:

                response = client.post(
                    self.url,
                    json=payload
                )

                response.raise_for_status()

                data = response.json()

                message = data.get(
                    "message",
                    {}
                )

                content = message.get(
                    "content"
                )

                if content and content.strip():

                    logger.info(
                        "Successfully received "
                        "response from Ollama."
                    )

                    return content.strip()

                logger.error(
                    "Ollama returned a response "
                    "without message content."
                )

                return (
                    "I couldn't generate an answer "
                    "from the repository evidence."
                )

        except httpx.ConnectError:

            logger.error(
                "Could not connect to Ollama at "
                "http://localhost:11434."
            )

            return (
                "I couldn't connect to the local Ollama "
                "service. Please make sure Ollama is running."
            )

        except httpx.TimeoutException:

            logger.error(
                "Ollama request timed out after "
                "120 seconds for model '%s'.",
                self.model
            )

            return (
                "The local AI model request timed out. "
                "Please verify Ollama system resource usage."
            )

        except httpx.HTTPError as exc:

            logger.error(
                "Ollama request failed: %s",
                exc
            )

            return (
                "The local AI model could not "
                "process the request."
            )


class OpenAILLM(LLMProvider):

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini"
    ):
        self.api_key = api_key
        self.model = model

    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> str:

        url = (
            "https://api.openai.com/v1/chat/completions"
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,

            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],

            "temperature": 0.1
        }

        with httpx.Client(
            timeout=45.0
        ) as client:

            response = client.post(
                url,
                headers=headers,
                json=payload
            )

            response.raise_for_status()

            data = response.json()

            return (
                data["choices"][0]
                ["message"]["content"]
            )


class GeminiLLM(LLMProvider):

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-1.5-flash"
    ):
        self.api_key = api_key
        self.model = model

    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> str:

        url = (
            "https://generativelanguage.googleapis.com/"
            f"v1beta/models/{self.model}:generateContent"
            f"?key={self.api_key}"
        )

        payload = {
            "system_instruction": {
                "parts": [
                    {
                        "text": system_prompt
                    }
                ]
            },

            "contents": [
                {
                    "parts": [
                        {
                            "text": user_prompt
                        }
                    ]
                }
            ],

            "generationConfig": {
                "temperature": 0.1
            }
        }

        with httpx.Client(
            timeout=45.0
        ) as client:

            response = client.post(
                url,
                json=payload
            )

            response.raise_for_status()

            data = response.json()

            candidates = data.get(
                "candidates",
                []
            )

            if (
                candidates
                and candidates[0]
                .get("content", {})
                .get("parts")
            ):

                return (
                    candidates[0]
                    ["content"]["parts"][0]
                    ["text"]
                )

            return (
                "I couldn't find enough evidence "
                "in the repository to answer "
                "this confidently."
            )


def get_llm_provider() -> LLMProvider:

    provider = settings.LLM_PROVIDER.lower()

    # ---------------------------------------------------------
    # Ollama
    # ---------------------------------------------------------

    if provider == "ollama":

        logger.info(
            "Using local Ollama LLM: "
            "qwen2.5-coder:7b"
        )

        return OllamaLLM(
            model="qwen2.5-coder:7b"
        )

    # ---------------------------------------------------------
    # OpenAI
    # ---------------------------------------------------------

    if (
        provider == "openai"
        and settings.OPENAI_API_KEY
    ):

        logger.info(
            "Using OpenAI LLM: %s",
            settings.OPENAI_MODEL
        )

        return OpenAILLM(
            settings.OPENAI_API_KEY,
            settings.OPENAI_MODEL
        )

    # ---------------------------------------------------------
    # Gemini
    # ---------------------------------------------------------

    if (
        provider == "gemini"
        and settings.GEMINI_API_KEY
    ):

        logger.info(
            "Using Gemini LLM: %s",
            settings.GEMINI_MODEL
        )

        return GeminiLLM(
            settings.GEMINI_API_KEY,
            settings.GEMINI_MODEL
        )

    # ---------------------------------------------------------
    # Mock fallback
    # ---------------------------------------------------------

    if provider != "mock":

        logger.warning(
            "LLM provider '%s' API key missing "
            "or invalid. Falling back to MockLLM.",
            provider
        )

    logger.info(
        "Using MockLLM."
    )

    return MockLLM()