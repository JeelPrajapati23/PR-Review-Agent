import asyncio
from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from app.agent import (
    InlineSuggestion,
    ReviewOutput,
    SpecialistFindings,
    _build_inline_comments,
    _build_panel_graph,
    _fetched_file_names,
    _format_review,
    _is_grounded,
)
from mcp_servers.code_server import scan_local_dependencies

# ---------------------------------------------------------------------------
# 1. scan_local_dependencies (AST parsing / import resolution)
# ---------------------------------------------------------------------------
# Real temp files via tmp_path rather than mocking builtins.open: the tool
# gates every read behind Path.is_dir()/.resolve()/.is_relative_to() checks
# that hit the real filesystem regardless of an open() mock, so a faithful
# mock would mean stubbing half of pathlib for no real benefit over writing
# a handful of throwaway files -- still fully offline and fast either way.


def test_scan_local_dependencies_resolves_absolute_and_relative_local_imports(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "main.py").write_text("x = 1\n")
    (tmp_path / "app" / "tasks.py").write_text("y = 2\n")
    (tmp_path / "app" / "consumer.py").write_text(
        "import app.main\n"
        "from . import tasks\n"
        "import os\n"
        "from collections import OrderedDict\n"
    )

    result = scan_local_dependencies(repo_path=str(tmp_path), file_path="app/consumer.py")

    assert result == "app/main.py\napp/tasks.py"
    assert "os" not in result
    assert "collections" not in result
    assert "OrderedDict" not in result


def test_scan_local_dependencies_filters_out_stdlib_only_imports(tmp_path):
    (tmp_path / "isolated.py").write_text("import os\nimport json\nfrom collections import OrderedDict\n")

    result = scan_local_dependencies(repo_path=str(tmp_path), file_path="isolated.py")

    assert result == "No local project dependencies found."


def test_scan_local_dependencies_reports_error_for_missing_repo_path(tmp_path):
    result = scan_local_dependencies(repo_path=str(tmp_path / "does-not-exist"), file_path="isolated.py")

    assert result.startswith("Error:")


def test_scan_local_dependencies_rejects_path_escaping_repo_root(tmp_path):
    result = scan_local_dependencies(repo_path=str(tmp_path), file_path="../outside.py")

    assert result.startswith("Error:")
    assert "escapes repo root" in result


# ---------------------------------------------------------------------------
# 2. Grounding and markdown formatting guardrails
# ---------------------------------------------------------------------------


def _fetch_call(call_id: str, file_path: str) -> dict:
    return {"name": "fetch_file_contents", "args": {"file_path": file_path}, "id": call_id}


def test_fetched_file_names_includes_only_successfully_fetched_files():
    messages = [
        AIMessage(content="", tool_calls=[_fetch_call("call_1", "app/main.py"), _fetch_call("call_2", "app/missing.py")]),
        ToolMessage(content="x = 1", tool_call_id="call_1"),
        ToolMessage(content="Error: file not found", tool_call_id="call_2"),
    ]

    assert _fetched_file_names(messages) == {"main.py"}


def test_fetched_file_names_ignores_non_fetch_tool_calls():
    messages = [
        AIMessage(
            content="",
            tool_calls=[{"name": "scan_local_dependencies", "args": {"file_path": "app/main.py"}, "id": "call_1"}],
        ),
        ToolMessage(content="app/tasks.py", tool_call_id="call_1"),
    ]

    assert _fetched_file_names(messages) == set()


def test_is_grounded_true_when_review_references_a_fetched_file():
    assert _is_grounded("An unhandled exception in main.py breaks the endpoint.", {"main.py"}) is True


def test_is_grounded_false_when_review_names_a_file_never_fetched():
    # Fabrication case: the model describes a file it never actually read.
    assert _is_grounded("SQL injection found in fabricated_module.py.", {"main.py"}) is False


def test_is_grounded_false_when_nothing_was_fetched():
    assert _is_grounded("Issue found in main.py.", set()) is False


def test_format_review_renders_all_sections_in_order():
    review = ReviewOutput(
        summary="Reviewed app/main.py for correctness.",
        performance_observations="None found.",
        edge_cases="None found.",
        security_correctness_issues="Hardcoded secret in app/main.py.",
        architectural_suggestions="None found.",
        validation_outcome="None found.",
    )

    assert _format_review(review) == (
        "## Summary\nReviewed app/main.py for correctness.\n\n"
        "## Performance Observations\nNone found.\n\n"
        "## Edge Cases\nNone found.\n\n"
        "## Security & Correctness Issues\nHardcoded secret in app/main.py.\n\n"
        "## Architectural Suggestions\nNone found.\n\n"
        "## Validation Outcome\nNone found."
    )


def test_build_inline_comments_renders_exact_suggestion_markdown():
    suggestion = InlineSuggestion(
        file_path="app/main.py",
        line=42,
        suggested_code="return sanitize(x)",
        comment="Sanitize user input before returning it.",
    )

    comments = _build_inline_comments([suggestion], fetched_files={"main.py"})

    assert comments == [
        {
            "path": "app/main.py",
            "line": 42,
            "side": "RIGHT",
            "body": "Sanitize user input before returning it.\n\n```suggestion\nreturn sanitize(x)\n```",
        }
    ]


def test_build_inline_comments_drops_suggestions_for_unfetched_files():
    suggestion = InlineSuggestion(
        file_path="app/fabricated.py", line=1, suggested_code="pass", comment="Fabricated fix."
    )

    assert _build_inline_comments([suggestion], fetched_files={"main.py"}) == []


# ---------------------------------------------------------------------------
# 3. Mock integration test for the sequential multi-agent graph topology
# ---------------------------------------------------------------------------


class _StubTool:
    def __init__(self, name: str):
        self.name = name


def test_panel_graph_runs_sequential_topology_and_populates_state():
    # Patches the two chokepoints where ChatGroq is actually constructed and
    # invoked (_run_specialist for each specialist, _synthesize for the
    # merge pass) rather than mocking langchain_groq.ChatGroq itself -- the
    # graph nodes only ever consume these functions' return values, so this
    # exercises the real topology/state-reducer wiring in _build_panel_graph
    # without coupling the test to create_react_agent's internal calling
    # conventions. No network access, no real LLM call, no real Redis.
    call_order: list[str] = []

    async def fake_run_specialist(persona_key, _prompt, _tools, _checkpointer, _base_thread_id, _task_message, _target_files):
        call_order.append(persona_key)
        if persona_key == "security":
            return {
                "findings": SpecialistFindings(
                    findings="SUGGESTION file=app/main.py line=5: validate(x) || Missing input validation",
                    validation_notes="None found.",
                ),
                "fetched_files": ["main.py"],
                "circuit_broken": False,
                "last_message": "security done",
            }
        return {
            "findings": SpecialistFindings(findings="O(n^2) loop in app/main.py", validation_notes="None found."),
            "fetched_files": ["main.py"],
            "circuit_broken": False,
            "last_message": "performance done",
        }

    fake_review = ReviewOutput(
        summary="Reviewed app/main.py.",
        performance_observations="O(n^2) loop in app/main.py.",
        edge_cases="None found.",
        security_correctness_issues="Missing input validation in app/main.py.",
        architectural_suggestions="None found.",
        validation_outcome="None found.",
    )

    async def fake_synthesize(security, performance):
        assert security is not None
        assert performance is not None
        return fake_review

    tools = [
        _StubTool("fetch_file_contents"),
        _StubTool("scan_local_dependencies"),
        _StubTool("run_validation_suite"),
    ]
    checkpointer = MemorySaver()

    async def _run():
        graph = _build_panel_graph(tools, checkpointer, "test-thread")
        return await graph.ainvoke(
            {"task_message": "Review PR #1", "target_files": ["app/main.py"]},
            config={"configurable": {"thread_id": "test-thread"}},
        )

    with patch("app.agent._run_specialist", side_effect=fake_run_specialist), patch(
        "app.agent._synthesize", side_effect=fake_synthesize
    ):
        result = asyncio.run(_run())

    # Topology: security_warden -> performance_architect -> synthesizer,
    # strictly in that order (the whole point of the sequential graph).
    assert call_order == ["security", "performance"]

    assert result["security_findings"].findings.startswith("SUGGESTION file=app/main.py")
    assert result["security_circuit_broken"] is False
    assert result["performance_findings"].findings == "O(n^2) loop in app/main.py"
    assert result["performance_circuit_broken"] is False
    assert result["final_review"] == fake_review
