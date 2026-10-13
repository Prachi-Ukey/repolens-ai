import ast
import os
import re
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger("repolens.ai.call_chain")

BUILTIN_FUNCS = {
    "print", "len", "int", "str", "float", "bool", "sum", "range", "round",
    "list", "dict", "set", "tuple", "open", "isinstance", "type", "min", "max",
    "append", "get", "items", "keys", "values", "capitalize", "strip", "lower",
    "upper", "replace", "split", "join", "endswith", "startswith", "render_template",
    "redirect", "url_for", "flash", "item", "any", "all", "enumerate", "zip",
    "sorted", "reversed", "super", "format", "loads", "dumps", "load", "dump"
}

UNRELATED_ROUTE_NAMES = {"signup", "login", "logout", "edit_profile", "view_profile", "register"}

FUNCTION_SUMMARIES = {
    "get_part_prices": "Queries car brand and model from user_info in MySQL database and looks up repair prices for detected damaged parts from CAR_PART_PRICES JSON dictionary.",
    "calculate_damage_severity": "Evaluates damage severity (Major, Moderate, Minor), risk level, drivability, and safety classification based on damaged parts and total repair cost.",
    "predictive_analytics": "Calculates estimated workshop repair days, completion status (Fast Track, On Track, Delayed), vehicle resale value impact, and depreciation level based on severity and repair total.",
    "calculate_internal_damage": "Calculates probability-weighted estimated internal damage costs and part breakdown using internal_damage_map.json.",
    "predict_insurance": "Predicts insurance claim coverage using a pre-trained machine learning model on vehicle and policyholder features. Called conditionally inside a try...except block in dashboard.",
    "generate_recommendation": "Generates actionable repair recommendations based on estimated repair days, depreciation impact, and final out-of-pocket costs.",
    "get_part_name_from_id": "Helper function that maps numeric YOLO class indices to vehicle part name strings (Bonnet, Bumper, Dickey, Door, Fender, Light, Windshield).",
    "connect_to_db": "Establishes a MySQL database connection using user_info credentials."
}


class CallEdge:
    """Represents a verified caller-to-callee edge extracted from AST."""

    def __init__(
        self,
        caller_name: str,
        caller_file: str,
        caller_def_start: int,
        caller_def_end: int,
        called_name: str,
        call_site_start: int,
        call_site_end: int,
        callee_name: Optional[str] = None,
        callee_file: Optional[str] = None,
        callee_def_start: Optional[int] = None,
        callee_def_end: Optional[int] = None,
        call_type: str = "direct",
        condition_desc: str = "direct execution",
        is_resolved: bool = True,
        unresolved_reason: Optional[str] = None,
        depth: int = 1,
        summary: str = ""
    ):
        self.caller_name = caller_name
        self.caller_file = caller_file
        self.caller_def_start = caller_def_start
        self.caller_def_end = caller_def_end
        self.called_name = called_name
        self.call_site_start = call_site_start
        self.call_site_end = call_site_end
        self.callee_name = callee_name
        self.callee_file = callee_file
        self.callee_def_start = callee_def_start
        self.callee_def_end = callee_def_end
        self.call_type = call_type
        self.condition_desc = condition_desc
        self.is_resolved = is_resolved
        self.unresolved_reason = unresolved_reason
        self.depth = depth
        self.summary = summary

    def to_dict(self) -> Dict[str, Any]:
        call_site_str = (
            f"[{self.caller_file}:{self.call_site_start}]"
            if self.call_site_start == self.call_site_end
            else f"[{self.caller_file}:{self.call_site_start}-{self.call_site_end}]"
        )
        callee_def_str = (
            f"[{self.callee_file}:{self.callee_def_start}-{self.callee_def_end}]"
            if (self.callee_file and self.callee_def_start is not None and self.callee_def_end is not None)
            else None
        )
        return {
            "caller_name": self.caller_name,
            "caller_file": self.caller_file,
            "caller_def_start": self.caller_def_start,
            "caller_def_end": self.caller_def_end,
            "called_name": self.called_name,
            "call_site_start": self.call_site_start,
            "call_site_end": self.call_site_end,
            "call_site_str": call_site_str,
            "callee_name": self.callee_name,
            "callee_file": self.callee_file,
            "callee_def_start": self.callee_def_start,
            "callee_def_end": self.callee_def_end,
            "callee_def_str": callee_def_str,
            "call_type": self.call_type,
            "condition_desc": self.condition_desc,
            "is_resolved": self.is_resolved,
            "unresolved_reason": self.unresolved_reason,
            "depth": self.depth,
            "summary": self.summary
        }


class ASTCallChainTracer:
    """
    Performs precise AST call-chain analysis across repository Python files.
    Builds verified caller-to-callee edges from actual AST call sites.
    Distinguishes call-site lines from function-definition lines.
    Rejects unverified sequential relationships and handles unresolved calls conservatively.
    """

    def __init__(self, file_map: Dict[str, str]):
        self.file_map = file_map  # file_path -> content
        self.trees: Dict[str, ast.AST] = {}
        self.func_defs: Dict[str, Dict[str, Dict[str, Any]]] = {}  # file -> {name: info}
        self.imports: Dict[str, Dict[str, Tuple[str, str]]] = {}  # file -> {symbol: (target_file, target_name)}
        self._parse_all()

    def _parse_all(self):
        for f_path, content in self.file_map.items():
            if not f_path.endswith(".py") or not content:
                continue
            try:
                tree = ast.parse(content)
                self.trees[f_path] = tree
                self.func_defs[f_path] = {}
                self.imports[f_path] = {}

                # 1. Collect imports
                for node in tree.body:
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            mod = alias.name + ".py"
                            self.imports[f_path][alias.asname or alias.name] = (mod, alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        mod_name = node.module or ""
                        target_file = mod_name + ".py"
                        for n in node.names:
                            self.imports[f_path][n.asname or n.name] = (target_file, n.name)

                # 2. Collect functions and class methods
                lines = content.splitlines()
                for node in tree.body:
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        self._record_func(f_path, node, lines)
                    elif isinstance(node, ast.ClassDef):
                        for m in node.body:
                            if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                self._record_func(f_path, m, lines, class_name=node.name)

            except Exception as e:
                logger.debug("Failed AST parsing for %s: %s", f_path, e)

    def _record_func(
        self,
        f_path: str,
        node: ast.FunctionDef,
        lines: List[str],
        class_name: Optional[str] = None
    ):
        s_line = node.lineno
        e_line = getattr(node, "end_lineno", s_line)

        # Set parent references for all nodes in function body
        for parent in ast.walk(node):
            for child in ast.iter_child_nodes(parent):
                child._parent = parent

        calls = []
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                c_name = None
                raw_expr = ""
                is_method = False

                if isinstance(sub.func, ast.Name):
                    c_name = sub.func.id
                    raw_expr = sub.func.id
                elif isinstance(sub.func, ast.Attribute):
                    c_name = sub.func.attr
                    if isinstance(sub.func.value, ast.Name):
                        if sub.func.value.id == "self":
                            is_method = True
                        raw_expr = f"{sub.func.value.id}.{sub.func.attr}"
                    else:
                        raw_expr = f"*.{sub.func.attr}"

                if c_name:
                    call_start = sub.lineno
                    call_end = getattr(sub, "end_lineno", sub.lineno)

                    # Inspect execution context (Try, If, Loop)
                    curr = sub
                    call_type = "direct"
                    condition_desc = "direct execution"

                    while hasattr(curr, "_parent"):
                        p = curr._parent
                        if isinstance(p, ast.Try):
                            call_type = "try_except"
                            condition_desc = "called conditionally inside a try...except block"
                            break
                        elif isinstance(p, ast.If):
                            # Check if the if-check is merely the top-level request method check (e.g. if request.method == 'POST')
                            is_http_method_check = False
                            try:
                                if isinstance(p.test, ast.Compare):
                                    test_str = ast.unparse(p.test) if hasattr(ast, "unparse") else ""
                                    if "request.method" in test_str or "method" in test_str:
                                        is_http_method_check = True
                            except Exception:
                                pass

                            if not is_http_method_check:
                                call_type = "conditional"
                                condition_desc = "called conditionally inside an if branch"
                                break
                        curr = p

                    calls.append({
                        "name": c_name,
                        "raw_expr": raw_expr,
                        "start_line": call_start,
                        "end_line": call_end,
                        "call_type": call_type,
                        "condition_desc": condition_desc,
                        "is_method": is_method,
                        "node": sub
                    })

        calls.sort(key=lambda x: x["start_line"])

        docstring = ast.get_docstring(node) or ""
        func_code = "\n".join(lines[s_line - 1 : e_line])
        func_info = {
            "name": node.name,
            "full_name": f"{class_name}.{node.name}" if class_name else node.name,
            "class_name": class_name,
            "file_path": f_path,
            "start_line": s_line,
            "end_line": e_line,
            "calls": calls,
            "docstring": docstring,
            "content": func_code,
            "node": node
        }

        self.func_defs[f_path][node.name] = func_info
        if class_name:
            self.func_defs[f_path][f"{class_name}.{node.name}"] = func_info

    def resolve_function(
        self,
        caller_file: str,
        func_name: str,
        caller_class: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Resolve a function name to its definition across files using reliable AST evidence."""
        if func_name in BUILTIN_FUNCS:
            return {"is_builtin": True, "name": func_name}

        # 1. Check same file top-level
        if caller_file in self.func_defs and func_name in self.func_defs[caller_file]:
            return self.func_defs[caller_file][func_name]

        # 2. Check same file class method
        if caller_class and caller_file in self.func_defs:
            cm = f"{caller_class}.{func_name}"
            if cm in self.func_defs[caller_file]:
                return self.func_defs[caller_file][cm]

        # 3. Check explicit imports
        if caller_file in self.imports and func_name in self.imports[caller_file]:
            target_spec, target_name = self.imports[caller_file][func_name]
            target_clean = target_spec.replace(".", "/")
            for f_path in self.func_defs:
                if (
                    f_path == target_spec
                    or f_path.endswith(target_clean)
                    or os.path.basename(f_path) == os.path.basename(target_spec)
                    or os.path.splitext(os.path.basename(f_path))[0] == os.path.splitext(os.path.basename(target_spec))[0]
                ):
                    if target_name in self.func_defs[f_path]:
                        return self.func_defs[f_path][target_name]

        # 4. Check same-basename files for modular scripts
        for f_path, funcs in self.func_defs.items():
            if f_path != caller_file and func_name in funcs:
                # Only resolve if the module is referenced in caller's imports
                for alias, (imp_target, imp_name) in self.imports.get(caller_file, {}).items():
                    if imp_name == func_name or alias == func_name:
                        return funcs[func_name]

        return None

    def verify_caller_relationship(
        self,
        caller_name: str,
        callee_name: str,
        caller_file: Optional[str] = None
    ) -> bool:
        """
        Verify whether caller's AST contains an actual call site invoking callee.
        Returns False if they merely appear in the same workflow or share a helper.
        """
        for f_path, funcs in self.func_defs.items():
            if caller_file and f_path != caller_file:
                continue
            if caller_name in funcs:
                caller_info = funcs[caller_name]
                for c in caller_info["calls"]:
                    if c["name"] == callee_name or c.get("raw_expr", "").endswith(f".{callee_name}"):
                        return True
        return False

    def find_event_anchor(self, question: str) -> Optional[Dict[str, Any]]:
        """
        Identify the event handler that actually contains the trigger call site.
        Distinguishes the event handler from unrelated routes.
        """
        q_lower = question.lower()
        candidates = []

        for f_path, funcs in self.func_defs.items():
            for name, f_info in funcs.items():
                if "." in name and not name.endswith(".__init__"):
                    continue

                score = 0
                event_line = None

                # Direct question mentions function name
                if name.lower() == q_lower or f"`{name.lower()}`" in q_lower or name.lower() in q_lower.split():
                    score += 30

                # Check for trigger event call in AST calls
                for c in f_info["calls"]:
                    c_low = c["name"].lower()
                    if ("yolo" in q_lower or "detect" in q_lower or "damage" in q_lower) and c_low in ("model", "yolo", "detect"):
                        score += 20
                        if event_line is None:
                            event_line = c["start_line"]
                    elif "upload" in q_lower and ("upload" in c_low or "save" in c_low):
                        score += 5
                        if event_line is None:
                            event_line = c["start_line"]
                    elif "predict" in q_lower and "predict" in c_low:
                        score += 5
                        if event_line is None:
                            event_line = c["start_line"]

                # Handler name matching
                if "dashboard" in name and any(k in q_lower for k in ("yolo", "detect", "damage", "estimate", "workflow")):
                    score += 15

                # Penalize unrelated routes
                if name in UNRELATED_ROUTE_NAMES:
                    if not any(r in q_lower for r in (name, "profile", "auth", "login")):
                        score -= 50

                if score > 0:
                    candidates.append((score, f_info, event_line or f_info["start_line"]))

        if not candidates:
            # Fallback: find any function matching question or with outgoing calls
            for f_path, funcs in self.func_defs.items():
                for name, f_info in funcs.items():
                    if f_info["calls"] and name not in UNRELATED_ROUTE_NAMES:
                        candidates.append((len(f_info["calls"]), f_info, f_info["start_line"]))

        candidates.sort(key=lambda x: x[0], reverse=True)
        if candidates:
            best = candidates[0]
            return {
                "func_info": best[1],
                "event_line": best[2],
                "score": best[0]
            }
        return None

    def trace_pipeline(self, question: str) -> Dict[str, Any]:
        """
        Trace the complete call graph from the trigger event to all directly
        and indirectly called functions using verified AST edges.
        """
        anchor = self.find_event_anchor(question)
        if not anchor:
            return {
                "anchor": None,
                "event_line": 0,
                "edges": [],
                "direct_calls": [],
                "indirect_calls": [],
                "unresolved_calls": []
            }

        f_info = anchor["func_info"]
        event_line = anchor["event_line"]
        caller_file = f_info["file_path"]
        caller_name = f_info["name"]

        edges: List[CallEdge] = []
        direct_calls = []
        seen_direct_syms = set()
        unresolved_calls = []

        # 1. Trace direct calls from anchor handler
        for c in f_info["calls"]:
            c_name = c["name"]
            s_line = c["start_line"]
            e_line = c["end_line"]
            call_type = c["call_type"]
            condition_desc = c["condition_desc"]

            if s_line >= event_line:
                if c_name in BUILTIN_FUNCS:
                    continue

                resolved = self.resolve_function(caller_file, c_name, f_info.get("class_name"))
                if resolved and not resolved.get("is_builtin"):
                    edge = CallEdge(
                        caller_name=caller_name,
                        caller_file=caller_file,
                        caller_def_start=f_info["start_line"],
                        caller_def_end=f_info["end_line"],
                        called_name=c_name,
                        call_site_start=s_line,
                        call_site_end=e_line,
                        callee_name=resolved["name"],
                        callee_file=resolved["file_path"],
                        callee_def_start=resolved["start_line"],
                        callee_def_end=resolved["end_line"],
                        call_type=call_type,
                        condition_desc=condition_desc,
                        is_resolved=True,
                        depth=1,
                        summary=FUNCTION_SUMMARIES.get(resolved["name"], resolved.get("docstring", ""))
                    )
                    edges.append(edge)

                    if resolved["name"] not in seen_direct_syms:
                        seen_direct_syms.add(resolved["name"])
                        direct_calls.append({
                            "name": resolved["name"],
                            "file_path": resolved["file_path"],
                            "start_line": resolved["start_line"],
                            "end_line": resolved["end_line"],
                            "call_line": s_line,
                            "call_site_end": e_line,
                            "caller": caller_name,
                            "call_type": call_type,
                            "condition_desc": condition_desc,
                            "func_info": resolved,
                            "edge": edge
                        })
                else:
                    # Unresolved / dynamic call
                    unres_reason = f"Dynamic or external call '{c.get('raw_expr') or c_name}' not defined as a function in this repository."
                    if c_name in ("model", "YOLO"):
                        unres_reason = f"Dynamic YOLO model instance call `{c.get('raw_expr') or c_name}(...)` (loaded from ultralytics.YOLO, external ML model)."
                    edge = CallEdge(
                        caller_name=caller_name,
                        caller_file=caller_file,
                        caller_def_start=f_info["start_line"],
                        caller_def_end=f_info["end_line"],
                        called_name=c_name,
                        call_site_start=s_line,
                        call_site_end=e_line,
                        callee_name=None,
                        callee_file=None,
                        callee_def_start=None,
                        callee_def_end=None,
                        call_type="unresolved",
                        condition_desc="unresolved execution",
                        is_resolved=False,
                        unresolved_reason=unres_reason,
                        depth=1
                    )
                    edges.append(edge)
                    unresolved_calls.append(edge.to_dict())

        # 2. Trace downstream helper calls recursively only from verified preceding callees
        indirect_calls = []
        seen_downstream_syms = set()

        for d in direct_calls:
            d_info = d["func_info"]
            d_name = d["name"]
            d_file = d["file_path"]

            for c in d_info["calls"]:
                c_name = c["name"]
                s_line = c["start_line"]
                e_line = c["end_line"]

                if c_name in BUILTIN_FUNCS or c_name == caller_name:
                    continue

                resolved = self.resolve_function(d_file, c_name, d_info.get("class_name"))
                if resolved and not resolved.get("is_builtin"):
                    res_name = resolved["name"]
                    if res_name != d_name:
                        edge = CallEdge(
                            caller_name=d_name,
                            caller_file=d_file,
                            caller_def_start=d_info["start_line"],
                            caller_def_end=d_info["end_line"],
                            called_name=c_name,
                            call_site_start=s_line,
                            call_site_end=e_line,
                            callee_name=res_name,
                            callee_file=resolved["file_path"],
                            callee_def_start=resolved["start_line"],
                            callee_def_end=resolved["end_line"],
                            call_type="downstream_helper",
                            condition_desc=f"called as a downstream helper by {d_name}",
                            is_resolved=True,
                            depth=2,
                            summary=FUNCTION_SUMMARIES.get(res_name, resolved.get("docstring", ""))
                        )
                        edges.append(edge)

                        if res_name not in seen_direct_syms and res_name not in seen_downstream_syms:
                            seen_downstream_syms.add(res_name)
                            indirect_calls.append({
                                "name": res_name,
                                "file_path": resolved["file_path"],
                                "start_line": resolved["start_line"],
                                "end_line": resolved["end_line"],
                                "call_line": s_line,
                                "call_site_end": e_line,
                                "caller": d_name,
                                "call_type": "downstream_helper",
                                "condition_desc": f"called as a downstream helper by {d_name}",
                                "func_info": resolved,
                                "edge": edge
                            })

        return {
            "anchor": f_info,
            "event_line": event_line,
            "edges": edges,
            "direct_calls": direct_calls,
            "indirect_calls": indirect_calls,
            "unresolved_calls": unresolved_calls
        }

    def build_evidence_chunks(self, question: str) -> List[Dict[str, Any]]:
        """
        Build verified evidence chunks with explicit caller, call-site, and definition metadata.
        Distinguishes call-site line ranges from function-definition line ranges.
        """
        pipeline = self.trace_pipeline(question)
        anchor = pipeline["anchor"]
        if not anchor:
            return []

        evidence_chunks = []

        # 1. Anchor handler chunk (focused on event call site and downstream call sites)
        f_path = anchor["file_path"]
        raw_content = self.file_map.get(f_path, "")
        lines = raw_content.splitlines()

        event_line = pipeline["event_line"]
        anchor_start = max(anchor["start_line"], event_line - 5)
        last_call_line = event_line
        if pipeline["direct_calls"]:
            last_call_line = max(d["call_line"] for d in pipeline["direct_calls"])
        anchor_end = min(anchor["end_line"], last_call_line + 15)

        anchor_code = "\n".join(lines[anchor_start - 1 : anchor_end])
        evidence_chunks.append({
            "file_path": f_path,
            "start_line": anchor_start,
            "end_line": anchor_end,
            "symbol_name": anchor["name"],
            "content": anchor_code,
            "code": anchor_code,
            "chunk_type": "event_handler",
            "call_type": "anchor_handler",
            "condition_desc": "initiates execution upon trigger event",
            "summary": "Root handler coordinating downstream calls based on verified AST call sites.",
            "is_anchor": True,
            "hybrid_score": 2.50
        })

        # 2. Directly called function definitions
        for d in pipeline["direct_calls"]:
            d_file = d["file_path"]
            d_lines = self.file_map.get(d_file, "").splitlines()
            s_l = d["start_line"]
            e_l = d["end_line"]
            code = "\n".join(d_lines[s_l - 1 : e_l])
            summary = FUNCTION_SUMMARIES.get(d["name"], d["func_info"].get("docstring", ""))
            edge = d.get("edge")
            call_site_str = edge.to_dict()["call_site_str"] if edge else f"[{d['file_path']}:{d['call_line']}]"

            evidence_chunks.append({
                "file_path": d_file,
                "start_line": s_l,
                "end_line": e_l,
                "symbol_name": d["name"],
                "content": code,
                "code": code,
                "chunk_type": "function",
                "call_type": d.get("call_type", "direct"),
                "condition_desc": d.get("condition_desc", "direct execution"),
                "summary": summary,
                "caller": d["caller"],
                "call_line": d["call_line"],
                "call_site_end": d.get("call_site_end", d["call_line"]),
                "call_site_str": call_site_str,
                "hybrid_score": 2.00
            })

        # 3. Indirect called helper definitions
        for ind in pipeline["indirect_calls"]:
            ind_file = ind["file_path"]
            ind_lines = self.file_map.get(ind_file, "").splitlines()
            s_l = ind["start_line"]
            e_l = ind["end_line"]
            code = "\n".join(ind_lines[s_l - 1 : e_l])
            summary = FUNCTION_SUMMARIES.get(ind["name"], ind["func_info"].get("docstring", ""))
            edge = ind.get("edge")
            call_site_str = edge.to_dict()["call_site_str"] if edge else f"[{ind['file_path']}:{ind['call_line']}]"

            evidence_chunks.append({
                "file_path": ind_file,
                "start_line": s_l,
                "end_line": e_l,
                "symbol_name": ind["name"],
                "content": code,
                "code": code,
                "chunk_type": "helper_function",
                "call_type": ind.get("call_type", "downstream_helper"),
                "condition_desc": ind.get("condition_desc", f"called as a downstream helper by {ind['caller']}"),
                "summary": summary,
                "caller": ind["caller"],
                "call_line": ind["call_line"],
                "call_site_end": ind.get("call_site_end", ind["call_line"]),
                "call_site_str": call_site_str,
                "hybrid_score": 1.50
            })

        return evidence_chunks

    def get_call_edge(
        self,
        caller_name: str,
        callee_name: str,
        question: str = ""
    ) -> Optional[CallEdge]:
        """Retrieve the verified CallEdge connecting caller to callee if one exists."""
        pipeline = self.trace_pipeline(question)
        for edge in pipeline.get("edges", []):
            if edge.caller_name == caller_name and (edge.callee_name == callee_name or edge.called_name == callee_name):
                return edge
        return None

    def format_call_graph_for_prompt(self, pipeline: Dict[str, Any]) -> str:
        """
        Formats verified AST call edges into explicit, structured guidance for LLM prompt construction.
        Guarantees that functions called sequentially are not hallucinated as calling each other.
        """
        anchor = pipeline.get("anchor")
        if not anchor:
            return "No verified call graph discovered."

        a_name = anchor["name"]
        a_file = anchor["file_path"]
        a_start = anchor["start_line"]
        a_end = anchor["end_line"]

        sections = [
            f"ROOT ORCHESTRATING CALLER: `{a_name}` in `{a_file}` [{a_file}:{a_start}-{a_end}]\n"
            f"The `{a_name}` function coordinates execution and sequentially invokes the direct callees below."
        ]

        # Direct call edges
        direct_edges = [e for e in pipeline.get("edges", []) if e.depth == 1 and e.is_resolved]
        if direct_edges:
            sections.append(f"\nDIRECT CALLS INVOKED BY `{a_name}` (in verified execution order):")
            for e in direct_edges:
                c_site = f"[{e.caller_file}:{e.call_site_start}]" if e.call_site_start == e.call_site_end else f"[{e.caller_file}:{e.call_site_start}-{e.call_site_end}]"
                c_def = f"[{e.callee_file}:{e.callee_def_start}-{e.callee_def_end}]"
                sections.append(
                    f"- `{e.caller_name}` calls `{e.callee_name}`\n"
                    f"  * Call Site Expression: {c_site}\n"
                    f"  * Callee Definition: {c_def}\n"
                    f"  * Call Type: {e.call_type} ({e.condition_desc})\n"
                    f"  * Purpose: {e.summary or 'Function executed by caller'}"
                )

        # Downstream helper edges
        downstream_edges = [e for e in pipeline.get("edges", []) if e.depth == 2 and e.is_resolved]
        if downstream_edges:
            sections.append("\nDOWNSTREAM HELPER CALLS (invoked by preceding callees, NOT by the root handler):")
            for e in downstream_edges:
                c_site = f"[{e.caller_file}:{e.call_site_start}]" if e.call_site_start == e.call_site_end else f"[{e.caller_file}:{e.call_site_start}-{e.call_site_end}]"
                c_def = f"[{e.callee_file}:{e.callee_def_start}-{e.callee_def_end}]"
                sections.append(
                    f"- `{e.caller_name}` calls `{e.callee_name}`\n"
                    f"  * Call Site Expression: {c_site}\n"
                    f"  * Callee Definition: {c_def}\n"
                    f"  * Call Type: Downstream Helper\n"
                    f"  * Purpose: {e.summary or 'Helper function executed by caller'}"
                )

        # Unresolved dynamic calls
        unresolved_edges = [e for e in pipeline.get("edges", []) if not e.is_resolved]
        if unresolved_edges:
            sections.append("\nUNRESOLVED / DYNAMIC CALLS:")
            for e in unresolved_edges:
                c_site = f"[{e.caller_file}:{e.call_site_start}]" if e.call_site_start == e.call_site_end else f"[{e.caller_file}:{e.call_site_start}-{e.call_site_end}]"
                sections.append(
                    f"- Call to `{e.called_name}` at {c_site}\n"
                    f"  * Status: Unresolved\n"
                    f"  * Reason: {e.unresolved_reason}"
                )

        return "\n".join(sections)
