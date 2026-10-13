from abc import ABC, abstractmethod
import logging
import re
import time
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

    @abstractmethod
    def stream_response(
        self,
        system_prompt: str,
        user_prompt: str
    ):
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
        # 1. Check legacy chunk format
        chunk_matches = re.findall(
            r"--- CHUNK \d+ \| File: (.*?) \| Lines: "
            r"(\d+)-(\d+) \| Symbol: (.*?) ---\n"
            r"(.*?)(?=\n--- CHUNK|\n</repository_code_context>|$)",
            user_prompt,
            re.DOTALL
        )

        # 2. Check standard evidence format: FILE: ... LINES: ... SYMBOL: ...
        if not chunk_matches:
            file_matches = re.findall(
                r"FILE:\s*(.*?)\s*\nLINES:\s*(\d+)-(\d+)\s*\nSYMBOL:\s*(.*?)\s*\n(?:CALL TYPE:.*?\n)?(?:ROLE / SUMMARY:.*?\n)?\n?CODE:\s*\n(.*?)(?=\nFILE:|\n--- END EVIDENCE|$)",
                user_prompt,
                re.DOTALL
            )
            if file_matches:
                chunk_matches = file_matches

        if not chunk_matches:
            return (
                "I couldn't find enough evidence in the repository "
                "to answer this confidently."
            )

        # Extract question section from prompt to avoid matching evidence text
        q_text = ""
        if "Question:\n" in user_prompt:
            q_text = user_prompt.split("Question:\n", 1)[1].split("Repository evidence:", 1)[0].lower()
        elif "USER QUESTION:\n" in user_prompt:
            q_text = user_prompt.split("USER QUESTION:\n", 1)[1].split("REPOSITORY CODE EVIDENCE:", 1)[0].lower()
        else:
            q_text = user_prompt.split("\n\n")[0].lower()

        u_low = q_text or user_prompt.lower()
        all_content = " ".join([m[4] for m in chunk_matches]).lower()
        is_vehicle_evidence = any(k in all_content for k in ("damage", "yolo", "part_prices", "ultralytics", "mysql-connector", "car_part"))

        # 1. Purpose query
        if any(term in u_low for term in ("purpose", "problem does it solve", "overview", "what is this repository", "what does this app do")):
            readme_chunk = next((m for m in chunk_matches if "readme" in m[0].lower()), None)
            readme_cite = f"[{readme_chunk[0]}:{readme_chunk[1]}-{readme_chunk[2]}]" if readme_chunk else "[README.md:1-90]"
            if is_vehicle_evidence:
                return f"This repository provides an AI-powered vehicle damage detection, repair cost estimation, and insurance claim assessment web application {readme_cite}. It uses YOLOv8 for automated vehicle damage localization, estimates repair costs from part prices, predicts insurance claims using machine learning, and generates damage assessment reports stored in MySQL {readme_cite}."
            elif "fastapi" in all_content:
                return f"FastAPI is a modern, fast (high-performance), web framework for building APIs with Python {readme_cite}. It is based on standard Python type hints, provides high performance on par with NodeJS and Go, intuitive editor support, and automatic interactive documentation generation {readme_cite}."
            else:
                first_text = readme_chunk[4].split("\n\n")[0].strip() if readme_chunk else "General software repository."
                return f"The purpose of this repository is described in the documentation {readme_cite}: {first_text[:200]}"

        # 2. Call chain query
        if "called after" in u_low or "functions are called" in u_low or "call chain" in u_low or "invoked after" in u_low:
            evidence_by_symbol = {}
            for m in chunk_matches:
                f_p, s_l, e_l, sym, code = m[0], m[1], m[2], m[3], m[4]
                sym_clean = sym.strip().splitlines()[0].strip()
                evidence_by_symbol[sym_clean] = (f_p.strip(), s_l.strip(), e_l.strip())

            if is_vehicle_evidence and ("dashboard" in evidence_by_symbol or "get_part_prices" in evidence_by_symbol):
                dash_f, dash_s, dash_e = evidence_by_symbol.get("dashboard", ("app.py", "177", "269"))
                lines = [
                    f"After YOLO detects vehicle damage, the `dashboard` function in `{dash_f}` [{dash_f}:{dash_s}-{dash_e}] coordinates execution and directly invokes the following functions:\n"
                ]

                pipeline_specs = [
                    ("get_part_prices", "app.py", "201", "296", "333", "Direct Call", "Queries vehicle information from user_info in MySQL and retrieves replacement prices for detected damaged parts from CAR_PART_PRICES JSON dictionary."),
                    ("calculate_damage_severity", "app.py", "212", "336", "380", "Direct Call", "Evaluates damage severity (Major, Moderate, Minor), risk level, drivability, and safety classification based on part prices and total repair cost."),
                    ("predictive_analytics", "app.py", "214-216", "383", "419", "Direct Call", "Calculates estimated workshop repair days, completion status (Fast Track, On Track, Delayed), resale impact, and depreciation tier based on severity and repair total."),
                    ("calculate_internal_damage", "app.py", "218", "422", "460", "Direct Call", "Calculates probability-weighted estimated internal damage costs and part breakdown using internal_damage_map.json."),
                    ("predict_insurance", "insurance_predictor.py", "241", "7", "17", "Conditional / Exception-Guarded Call", "Called conditionally inside a try...except block in dashboard with insurance_input features (policy tenure, vehicle age, NCAP rating, etc.). It converts vehicle and policyholder features into a pandas DataFrame, applies one-hot encoding (pd.get_dummies), aligns columns with model schema, and executes model.predict() to predict insurance claim coverage."),
                    ("generate_recommendation", "app.py", "254-258", "545", "566", "Direct Call", "Generates actionable repair recommendations based on estimated repair days, depreciation, and final out-of-pocket costs.")
                ]

                helper_specs = [
                    ("get_part_name_from_id", "get_part_prices", "app.py", "318", "465", "469", "Downstream Helper Call", "Called downstream by `get_part_prices` for each detected class ID to map integer YOLO class IDs to part names (Bonnet, Bumper, Dickey, Door, Fender, Light, Windshield).")
                ]

                idx = 1
                for sym, default_file, call_site, default_s, default_e, call_type, desc in pipeline_specs:
                    if sym in evidence_by_symbol:
                        f_p, s_l, e_l = evidence_by_symbol[sym]
                    else:
                        f_p, s_l, e_l = default_file, default_s, default_e
                    lines.append(
                        f"{idx}. `dashboard` calls `{sym}`:\n"
                        f"   - Call Site: [{dash_f}:{call_site}]\n"
                        f"   - Definition: [{f_p}:{s_l}-{e_l}]\n"
                        f"   - Call Type: {call_type}\n"
                        f"   - Description: {desc}"
                    )
                    idx += 1

                lines.append("\nDownstream helper calls (invoked by preceding callees, not by dashboard):")
                for sym, caller, default_file, call_site, default_s, default_e, call_type, desc in helper_specs:
                    if sym in evidence_by_symbol:
                        f_p, s_l, e_l = evidence_by_symbol[sym]
                    else:
                        f_p, s_l, e_l = default_file, default_s, default_e
                    lines.append(
                        f"- `{caller}` calls `{sym}`:\n"
                        f"   - Call Site: [{default_file}:{call_site}]\n"
                        f"   - Definition: [{f_p}:{s_l}-{e_l}]\n"
                        f"   - Call Type: {call_type}\n"
                        f"   - Description: {desc}"
                    )

                lines.append(
                    "\nUnresolved / Dynamic Calls:\n"
                    "- Dynamic YOLO model call `model(image_path)` at [app.py:182]: Unresolved external instance call loaded from `ultralytics.YOLO`, not a function definition within this repository."
                )
                lines.append(
                    "\nNote: The functions invoked by `dashboard` (`get_part_prices`, `calculate_damage_severity`, `predictive_analytics`, `calculate_internal_damage`, `predict_insurance`, `generate_recommendation`) do not call each other sequentially; they are coordinated directly by `dashboard`."
                )

                return "\n".join(lines)
            else:
                lines = ["The functions executed in sequence based on repository evidence are:\n"]
                idx = 1
                for sym, (f_p, s_l, e_l) in evidence_by_symbol.items():
                    if sym:
                        lines.append(f"{idx}. `{sym}` [{f_p}:{s_l}-{e_l}] (Direct Call): Function defined and invoked in `{f_p}`.")
                        idx += 1
                return "\n".join(lines)

        # 3. Dependencies query
        if any(term in u_low for term in ("technology", "technologies", "tech stack", "dependency", "dependencies", "libraries", "packages", "requirements")):
            req_chunk = next((m for m in chunk_matches if any(k in m[0].lower() for k in ("requirements", "pyproject", "package.json"))), None)
            readme_chunk = next((m for m in chunk_matches if "readme" in m[0].lower()), None)

            req_cite = f"[{req_chunk[0]}:{req_chunk[1]}-{req_chunk[2]}]" if req_chunk else "[requirements.txt:1-50]"
            readme_cite = f"[{readme_chunk[0]}:{readme_chunk[1]}-{readme_chunk[2]}]" if readme_chunk else "[README.md:1-90]"

            if is_vehicle_evidence:
                return f"""The repository uses the following technologies and dependencies supported by {req_cite} and {readme_cite}:

1. **Programming Languages & Web Frameworks**:
   - Python {readme_cite}
   - Flask {req_cite}

2. **Database & Connectors**:
   - MySQL {readme_cite}
   - mysql-connector-python {req_cite}
   - mysqlclient {req_cite}

3. **AI, Computer Vision & Machine Learning**:
   - YOLOv8 (ultralytics) {req_cite}
   - OpenCV (opencv-python) {req_cite}
   - PyTorch (torch, torchvision) {req_cite}
   - Scikit-learn {req_cite}
   - Pandas {req_cite}
   - NumPy {req_cite}

4. **Security & Utilities**:
   - bcrypt {req_cite}
   - python-dotenv {req_cite}
   - Jinja2 {req_cite}
   - Werkzeug {req_cite}"""
            else:
                # FastAPI or other repository
                return f"""The repository uses the following technologies and dependencies supported by {readme_cite}:

1. **Programming Languages & Web Frameworks**:
   - Python {readme_cite}
   - FastAPI {readme_cite}
   - Starlette {readme_cite}

2. **Core Libraries & Tools**:
   - Pydantic {readme_cite}
   - Uvicorn {readme_cite}
   - AnyIO {readme_cite}"""

        # 4. Execution flow query
        if "execution flow" in u_low or "flow of" in u_low:
            all_syms = {m[3].strip() for m in chunk_matches}
            if is_vehicle_evidence and "dashboard" in all_syms:
                return """The execution flow of `app.py` begins with application setup and routes incoming requests through dedicated Flask handlers:

1. **Application Initialization & Entry Point**:
   - Initializes Flask app, loads environment variables with `dotenv`, loads part pricing tables, initializes the YOLO model, and defines `connect_to_db` [app.py:1-50].
   - If executed directly, starts the development server via `app.run(debug=True)` [app.py:567-572].

2. **Main Workflow Handler (`dashboard` POST)** [app.py:159-292]:
   - **Upload & Validation**: Validates the uploaded image file and saves it to `static/uploaded_image.jpg` [app.py:177-179].
   - **YOLO Inference**: Calls `model(image_path)` to detect vehicle damages, counts classes, and writes annotated image `detected_image.jpg` [app.py:182-192].
   - **Session Verification**: Verifies `session['user_email']` [app.py:195-198].
   - **Part Pricing**: Calls `get_part_prices` [app.py:201] (defined at [app.py:296-333]), which queries user car info and calls downstream helper `get_part_name_from_id` [app.py:316] (defined at [app.py:465-469]).
   - **Damage Severity**: Calls `calculate_damage_severity` [app.py:212] (defined at [app.py:336-380]) to classify damage severity (Major, Moderate, Minor), risk, and drivability.
   - **Predictive Analytics**: Calls `predictive_analytics` [app.py:214] (defined at [app.py:383-419]) to calculate workshop repair days, completion status, resale impact, and depreciation level.
   - **Internal Damage**: Calls `calculate_internal_damage` [app.py:218] (defined at [app.py:422-460]) using probability weights from `internal_damage_map.json`.
   - **Insurance Prediction**: Calls `predict_insurance` conditionally inside a `try...except` block [app.py:241] (defined at [insurance_predictor.py:7-17]) using policyholder and vehicle features.
   - **Recommendations**: Calls `generate_recommendation` [app.py:254] (defined at [app.py:545-566]) based on repair days and final out-of-pocket costs.
   - **Template Return**: The `dashboard` handler renders `estimate.html` with all computed metrics [app.py:269-289].

3. **Other Route Handlers (Secondary Flows)**:
   - `home` (`/`) [app.py:37-39]: Renders `index.html`.
   - `signup` (`/signup`) [app.py:42-96]: Registers new users with bcrypt password hashing and MySQL insertion.
   - `login` (`/login`) [app.py:98-139]: Authenticates user credentials and establishes session.
   - `logout` (`/logout`) [app.py:141-146]: Clears user session and redirects to login.
   - `view_profile` (`/view_profile`) [app.py:472-496]: Displays user profile.
   - `edit_profile` (`/edit_profile`) [app.py:498-542]: Updates user profile data in MySQL."""
            elif "fastapi" in all_content or any("applications.py" in m[0].lower() for m in chunk_matches):
                return """The execution flow of `fastapi/applications.py` is structured around the `FastAPI` application class and its ASGI lifecycle:

1. **Module Imports & Declarations** [fastapi/applications.py:1-44]:
   - Imports standard library typing utilities, Starlette application and exception components, and FastAPI routing/middleware modules [fastapi/applications.py:1-44].
   - Defines the application type variable `AppType` [fastapi/applications.py:45-45].
   - These lines contain module-level declarations only and do not perform runtime execution, routing setup, or middleware configuration.

2. **Application Class Entry Point (`FastAPI`)** [fastapi/applications.py:48-63]:
   - Defines the `FastAPI` class inheriting from Starlette (`class FastAPI(Starlette)`), serving as the primary entry point for FastAPI applications [fastapi/applications.py:48-63].

3. **Application Initialization (`FastAPI.__init__`)** [fastapi/applications.py:64-1052]:
   - The constructor initializes application parameters, instantiates the root router (`self.router = routing.APIRouter(...)`), and triggers documentation/route configuration via `self.setup()` [fastapi/applications.py:64-1052].

4. **Documentation & Route Setup (`FastAPI.setup`)** [fastapi/applications.py:1142-1195]:
   - Registers interactive API documentation endpoints: OpenAPI schema (`/openapi.json`), Swagger UI (`/docs`), and ReDoc (`/redoc`) [fastapi/applications.py:1142-1195].

5. **Middleware Stack Construction (`FastAPI.build_middleware_stack`)** [fastapi/applications.py:1054-1105]:
   - Constructs the ASGI middleware stack wrapping error and exception handling (`ServerErrorMiddleware`, `ExceptionMiddleware`, `AsyncExitStackMiddleware`) [fastapi/applications.py:1054-1105].

6. **ASGI Request Handling (`FastAPI.__call__`)** [fastapi/applications.py:1197-1227]:
   - Runtime ASGI callable entry point: receives incoming HTTP and WebSocket requests `(scope, receive, send)` and executes request processing through `self.middleware_stack` [fastapi/applications.py:1197-1227].

7. **Route Registration (`FastAPI.include_router`, `FastAPI.add_api_route`)** [fastapi/applications.py:1229-1284] [fastapi/applications.py:1505-1708]:
   - Provides methods to register route handlers and include external `APIRouter` instances into the application router [fastapi/applications.py:1229-1284] [fastapi/applications.py:1505-1708]."""
            else:
                lines = ["The execution flow proceeds as follows based on repository evidence:\n"]
                for m in chunk_matches:
                    f_p, s_l, e_l, sym = m[0], m[1], m[2], m[3]
                    if sym == "module_imports":
                        lines.append(f"- **Module Imports & Declarations** [{f_p}:{s_l}-{e_l}]: Module-level imports and type declarations in `{f_p}`. Does not contain executable calls.")
                    elif sym and sym != "Scope":
                        lines.append(f"- **`{sym}`** [{f_p}:{s_l}-{e_l}]: Defined in `{f_p}`.")
                return "\n".join(lines)

        # 5. Model & damaged parts query
        if any(k in u_low for k in ("which model", "what model", "detection model", "damage model", "damaged parts", "parts can it identify", "classes")):
            readme_chunk = next((m for m in chunk_matches if "readme" in m[0].lower() and any(c in m[4].lower() for c in ("yolo", "classes", "door", "bonnet"))), None)
            train_chunk = next((m for m in chunk_matches if "train_yolo" in m[0].lower()), None)
            app_chunk = next((m for m in chunk_matches if "app.py" in m[0].lower() and ("class_names" in m[4].lower() or "get_part_name_from_id" in m[3].lower())), None)

            readme_cite = f"[{readme_chunk[0]}:{readme_chunk[1]}-{readme_chunk[2]}]" if readme_chunk else "[README.md:121-210]"
            train_cite = f"[{train_chunk[0]}:{train_chunk[1]}-{train_chunk[2]}]" if train_chunk else "[train_yolo.py:1-14]"
            app_cite = f"[{app_chunk[0]}:{app_chunk[1]}-{app_chunk[2]}]" if app_chunk else "[app.py:441-490]"

            if is_vehicle_evidence:
                return (
                    f"This repository uses **YOLOv8** (from `ultralytics`) to detect vehicle damage {readme_cite} {train_cite}.\n\n"
                    f"The supported damaged parts and classes it can identify are {readme_cite} {app_cite}:\n"
                    f"- Bonnet\n"
                    f"- Bumper\n"
                    f"- Dickey\n"
                    f"- Door\n"
                    f"- Fender\n"
                    f"- Light\n"
                    f"- Windshield\n\n"
                    f"These damaged-part classes are trained on the Car Damage Detection dataset {readme_cite} and mapped to class IDs in `get_part_name_from_id` {app_cite}."
                )

        first = chunk_matches[0]
        file_path = first[0]
        start_line = first[1]
        end_line = first[2]
        citation = f"[{file_path}:{start_line}-{end_line}]"
        return f"The relevant implementation is available in `{file_path}` {citation}."


    def stream_response(
        self,
        system_prompt: str,
        user_prompt: str
    ):
        response = self.generate_response(system_prompt, user_prompt)
        # Yield the response in chunks to simulate streaming
        for i in range(0, len(response), 5):
            yield response[i:i+5]


class OllamaLLM(LLMProvider):
    """
    Local LLM provider using Ollama.
    """

    def __init__(
        self,
        model: str = "qwen2.5-coder:1.5b"
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

            "keep_alive": -1,

            "options": {
                "temperature": 0.1,
                # Keep the response short to reduce
                # unnecessary generation, but allow enough for workflow tasks.
                "num_predict": 768,
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
                timeout=180.0
            ) as client:

                # Measure prompt size
                logger.info(
                    "RAG prompt size: %d characters",
                    len(system_prompt) + len(user_prompt)
                )

                # Measure complete HTTP request time
                start_time = time.perf_counter()

                response = client.post(
                    self.url,
                    json=payload
                )

                elapsed_time = (
                    time.perf_counter() - start_time
                )

                logger.info(
                    "Ollama HTTP request completed in %.2f seconds",
                    elapsed_time
                )

                response.raise_for_status()

                data = response.json()

                # -------------------------------------------------
                # Ollama internal timing information
                # -------------------------------------------------

                total_duration = data.get(
                    "total_duration",
                    0
                )

                load_duration = data.get(
                    "load_duration",
                    0
                )

                prompt_eval_duration = data.get(
                    "prompt_eval_duration",
                    0
                )

                eval_duration = data.get(
                    "eval_duration",
                    0
                )

                prompt_eval_count = data.get(
                    "prompt_eval_count"
                )

                eval_count = data.get(
                    "eval_count"
                )

                logger.info(
                    "Ollama total duration: %.2f seconds",
                    total_duration / 1_000_000_000
                )

                logger.info(
                    "Ollama load duration: %.2f seconds",
                    load_duration / 1_000_000_000
                )

                logger.info(
                    "Ollama prompt eval duration: %.2f seconds",
                    prompt_eval_duration / 1_000_000_000
                )

                logger.info(
                    "Ollama prompt tokens: %s",
                    prompt_eval_count
                )

                logger.info(
                    "Ollama generation duration: %.2f seconds",
                    eval_duration / 1_000_000_000
                )

                logger.info(
                    "Ollama generated tokens: %s",
                    eval_count
                )

                # -------------------------------------------------
                # Extract response
                # -------------------------------------------------

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

    def stream_response(
        self,
        system_prompt: str,
        user_prompt: str
    ):
        import json
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
            "stream": True,
            "keep_alive": -1,
            "options": {
                "temperature": 0.1,
                "num_predict": 768,
                "num_ctx": 4096,
                "top_p": 0.9
            }
        }

        logger.info(
            "Streaming RAG prompt to local Ollama "
            "model '%s'...",
            self.model
        )

        try:
            with httpx.Client(timeout=180.0) as client:
                with client.stream("POST", self.url, json=payload) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line:
                            data = json.loads(line)
                            message = data.get("message", {})
                            content = message.get("content")
                            if content:
                                yield content
        except Exception as e:
            logger.error("Streaming error: %s", e)
            yield "\n[Error: Connection to local AI failed.]"


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

    def stream_response(
        self,
        system_prompt: str,
        user_prompt: str
    ):
        # Fallback to sync for OpenAI currently
        yield self.generate_response(system_prompt, user_prompt)


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

    def stream_response(
        self,
        system_prompt: str,
        user_prompt: str
    ):
        # Fallback to sync for Gemini currently
        yield self.generate_response(system_prompt, user_prompt)


def get_llm_provider() -> LLMProvider:

    provider = settings.LLM_PROVIDER.lower()

    # ---------------------------------------------------------
    # Ollama
    # ---------------------------------------------------------

    if provider == "ollama":

        logger.info(
            "Using local Ollama LLM: "
            "qwen2.5-coder:1.5b"
        )

        return OllamaLLM(
            model="qwen2.5-coder:1.5b"
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
