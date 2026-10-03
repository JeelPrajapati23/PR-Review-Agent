import asyncio
from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from app.agent import (
    InlineSuggestion,
    ReviewOutput,
    SpecialistFindings,
    _build_inline_comments,
    _merge_batch_outcomes,
    _plan_batches,
    _restore_indentation,
    _build_panel_graph,
    _combine_fetched_files,
    _fetched_file_names,
    _format_review,
    _is_grounded,
)
from mcp_servers.code_server import fetch_file_contents, scan_local_dependencies

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


def test_combine_fetched_files_includes_diff_shown_files_without_a_fetch_call():
    # No fetch_file_contents call at all -- the file was reviewed via its
    # diff, embedded directly in the task message (see _build_task_messages).
    messages = [AIMessage(content="Reviewed the diff, no issues found.")]

    assert _combine_fetched_files(messages, ["main.py"]) == {"main.py"}


def test_combine_fetched_files_unions_diff_shown_and_actually_fetched():
    messages = [
        AIMessage(content="", tool_calls=[_fetch_call("call_1", "app/helper.py")]),
        ToolMessage(content="def helper(): ...", tool_call_id="call_1"),
    ]

    assert _combine_fetched_files(messages, ["main.py"]) == {"main.py", "helper.py"}


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

    comments = _build_inline_comments(
        [suggestion], fetched_files={"main.py"}, commentable_lines={"app/main.py": {42: "return x"}}
    )

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

    assert (
        _build_inline_comments(
            [suggestion], fetched_files={"main.py"}, commentable_lines={"app/fabricated.py": {1: "pass"}}
        )
        == []
    )


def test_build_inline_comments_drops_suggestions_outside_the_diff():
    # main.py was fetched (passes the fabrication guard), but the suggested
    # line isn't part of the PR's actual diff -- GitHub would 422 the whole
    # review if this were posted, so it must be dropped before that happens.
    suggestion = InlineSuggestion(
        file_path="app/main.py", line=99, suggested_code="pass", comment="Not part of this PR's diff."
    )

    assert (
        _build_inline_comments(
            [suggestion], fetched_files={"main.py"}, commentable_lines={"app/main.py": {42: "return x"}}
        )
        == []
    )


# Shapes observed live on a real PR: the model returned a fix for an indented
# line inside a function body with its leading whitespace stripped, which
# GitHub's "Commit suggestion" would have applied as an IndentationError.
_ORIGINAL_QUERY_LINE = '    query = "SELECT * FROM users WHERE username = \'" + username + "\'"'


def test_restore_indentation_fixes_single_stripped_line():
    assert (
        _restore_indentation('query = "SELECT * FROM users WHERE username = ?"', _ORIGINAL_QUERY_LINE)
        == '    query = "SELECT * FROM users WHERE username = ?"'
    )


def test_restore_indentation_fixes_only_first_line_when_rest_already_indented():
    suggested = 'query = "SELECT * FROM users WHERE username = ?"\n    return conn.execute(query, (username,)).fetchone()'

    assert _restore_indentation(suggested, _ORIGINAL_QUERY_LINE) == (
        '    query = "SELECT * FROM users WHERE username = ?"\n'
        "    return conn.execute(query, (username,)).fetchone()"
    )


def test_restore_indentation_pads_uniformly_dedented_block():
    suggested = "if user is None:\n    return None"

    assert _restore_indentation(suggested, "    if not user:") == "    if user is None:\n        return None"


def test_restore_indentation_leaves_correct_suggestions_untouched():
    suggested = '    query = "SELECT 1"\n    return query'

    assert _restore_indentation(suggested, _ORIGINAL_QUERY_LINE) == suggested
    assert _restore_indentation("print('x')", "print('working..?')") == "print('x')"


def test_build_inline_comments_restores_stripped_indentation():
    suggestion = InlineSuggestion(
        file_path="demo.py",
        line=2,
        suggested_code='query = "SELECT * FROM users WHERE username = ?"',
        comment="Use a parameterized query.",
    )

    comments = _build_inline_comments(
        [suggestion], fetched_files={"demo.py"}, commentable_lines={"demo.py": {2: _ORIGINAL_QUERY_LINE}}
    )

    assert '```suggestion\n    query = "SELECT * FROM users WHERE username = ?"\n```' in comments[0]["body"]


_GET_USER_LINES = {
    1: "def get_user(conn, username):",
    2: _ORIGINAL_QUERY_LINE,
    3: "    return conn.execute(query).fetchone()",
    4: "",
    5: "",
    6: 'print("working..?")',
}


def test_build_inline_comments_extends_range_over_rewritten_following_lines():
    # Real shape from a live PR: a two-line rewrite anchored only to line 2.
    # Posted as-is, committing it would leave the original line-3 return below
    # the new one; it must span lines 2-3 instead.
    suggestion = InlineSuggestion(
        file_path="demo.py",
        line=2,
        suggested_code='query = "SELECT * FROM users WHERE username = ?"\n    return conn.execute(query, (username,)).fetchone()',
        comment="Use a parameterized query.",
    )

    [comment] = _build_inline_comments(
        [suggestion], fetched_files={"demo.py"}, commentable_lines={"demo.py": _GET_USER_LINES}
    )

    assert (comment["start_line"], comment["start_side"], comment["line"]) == (2, "RIGHT", 3)


def test_build_inline_comments_keeps_genuine_insertion_single_line():
    # The extra line doesn't rewrite line 3 (different leading token), so this
    # is an insertion: anchored to line 2 alone, with line 3 kept.
    suggestion = InlineSuggestion(
        file_path="demo.py",
        line=2,
        suggested_code='    if not username:\n        raise ValueError("username required")',
        comment="Validate input.",
    )

    [comment] = _build_inline_comments(
        [suggestion], fetched_files={"demo.py"}, commentable_lines={"demo.py": _GET_USER_LINES}
    )

    assert comment["line"] == 2 and "start_line" not in comment


def test_build_inline_comments_honors_explicit_end_line():
    suggestion = InlineSuggestion(
        file_path="demo.py",
        line=2,
        end_line=3,
        suggested_code="    return conn.execute(\"SELECT * FROM users WHERE username = ?\", (username,)).fetchone()",
        comment="Inline the parameterized query.",
    )

    [comment] = _build_inline_comments(
        [suggestion], fetched_files={"demo.py"}, commentable_lines={"demo.py": _GET_USER_LINES}
    )

    assert (comment["start_line"], comment["line"]) == (2, 3)


def test_build_inline_comments_drops_overlapping_duplicate_suggestions():
    # Real shape from a live PR: both specialists proposed the same 2-3 fix,
    # and both were posted. The first (Security Warden's) must win; a fix on
    # a separate, non-overlapping line must survive.
    code = '    query = "SELECT * FROM orders WHERE id = ?"\n    return conn.execute(query, (order_id,)).fetchone()'
    security_fix = InlineSuggestion(file_path="demo.py", line=2, suggested_code=code, comment="Security fix.")
    performance_fix = InlineSuggestion(file_path="demo.py", line=3, suggested_code="    return None", comment="Overlaps.")
    separate_fix = InlineSuggestion(file_path="demo.py", line=6, suggested_code="# removed", comment="Separate.")

    comments = _build_inline_comments(
        [security_fix, performance_fix, separate_fix],
        fetched_files={"demo.py"},
        commentable_lines={"demo.py": _GET_USER_LINES},
    )

    assert [c["body"].split("\n")[0] for c in comments] == ["Security fix.", "Separate."]


def test_build_inline_comments_drops_range_crossing_a_hunk_gap():
    suggestion = InlineSuggestion(
        file_path="demo.py", line=2, end_line=4, suggested_code="    pass", comment="Spans a gap."
    )

    assert (
        _build_inline_comments(
            [suggestion], fetched_files={"demo.py"}, commentable_lines={"demo.py": {2: "    a", 4: "    b"}}
        )
        == []
    )


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

    security_fix = InlineSuggestion(
        file_path="app/main.py", line=5, suggested_code="validate(x)", comment="Missing input validation"
    )
    performance_fix = InlineSuggestion(
        file_path="app/main.py", line=12, suggested_code="items[:n]", comment="O(n^2) loop"
    )

    async def fake_run_specialist(
        persona_key, _prompt, _tools, _checkpointer, _base_thread_id, _task_messages, _target_files, _diff_shown_files
    ):
        call_order.append(persona_key)
        if persona_key == "security":
            return {
                "findings": SpecialistFindings(
                    findings="Missing input validation in app/main.py",
                    validation_notes="None found.",
                    suggested_fixes=[security_fix],
                ),
                "fetched_files": ["main.py"],
                "circuit_broken": False,
                "last_message": "security done",
            }
        return {
            "findings": SpecialistFindings(
                findings="O(n^2) loop in app/main.py",
                validation_notes="None found.",
                suggested_fixes=[performance_fix],
            ),
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
            {"task_messages": ["Review PR #1"], "target_files": ["app/main.py"], "diff_shown_files": []},
            config={"configurable": {"thread_id": "test-thread"}},
        )

    with patch("app.agent._run_specialist", side_effect=fake_run_specialist), patch(
        "app.agent._synthesize", side_effect=fake_synthesize
    ):
        result = asyncio.run(_run())

    # Topology: security_warden -> performance_architect -> synthesizer,
    # strictly in that order (the whole point of the sequential graph).
    assert call_order == ["security", "performance"]

    assert result["security_findings"].findings == "Missing input validation in app/main.py"
    assert result["security_circuit_broken"] is False
    assert result["performance_findings"].findings == "O(n^2) loop in app/main.py"
    assert result["performance_circuit_broken"] is False

    # inline_suggestions is assembled in synthesizer_node from each
    # specialist's own suggested_fixes, not left to whatever _synthesize
    # happened to return (fake_synthesize's return value started with an
    # empty inline_suggestions list -- synthesizer_node mutates that same
    # object in place, so this also confirms the other fields pass through
    # untouched).
    final_review = result["final_review"]
    assert final_review.inline_suggestions == [security_fix, performance_fix]
    assert final_review.summary == "Reviewed app/main.py."
    assert final_review.performance_observations == "O(n^2) loop in app/main.py."
    assert final_review.security_correctness_issues == "Missing input validation in app/main.py."


def test_panel_graph_parallel_runs_specialists_concurrently_then_synthesizes_once():
    # Each fake specialist waits for the other to have started. Under the
    # sequential topology this would time out; it only completes if both
    # are genuinely in flight at the same time.
    started = {"security": asyncio.Event(), "performance": asyncio.Event()}
    synth_calls: list[tuple] = []

    async def fake_run_specialist(
        persona_key, _prompt, _tools, _checkpointer, _base_thread_id, _task_messages, _target_files, _diff_shown_files
    ):
        started[persona_key].set()
        other = "performance" if persona_key == "security" else "security"
        await asyncio.wait_for(started[other].wait(), timeout=2)
        return {
            "findings": SpecialistFindings(
                findings=f"{persona_key} issue in app/main.py",
                validation_notes="None found.",
                suggested_fixes=[],
            ),
            "fetched_files": ["main.py"],
            "circuit_broken": False,
            "last_message": f"{persona_key} done",
        }

    async def fake_synthesize(security, performance):
        synth_calls.append((security, performance))
        return ReviewOutput(
            summary="Reviewed app/main.py.",
            performance_observations="None found.",
            edge_cases="None found.",
            security_correctness_issues="None found.",
            architectural_suggestions="None found.",
            validation_outcome="None found.",
        )

    tools = [_StubTool("fetch_file_contents")]

    async def _run():
        graph = _build_panel_graph(tools, MemorySaver(), "test-thread", parallel=True)
        return await graph.ainvoke(
            {"task_messages": ["Review PR #1"], "target_files": ["app/main.py"], "diff_shown_files": []},
            config={"configurable": {"thread_id": "test-thread"}},
        )

    with patch("app.agent._run_specialist", side_effect=fake_run_specialist), patch(
        "app.agent._synthesize", side_effect=fake_synthesize
    ):
        result = asyncio.run(_run())

    # Synthesizer joins both branches: called exactly once, with both reports.
    assert len(synth_calls) == 1
    security, performance = synth_calls[0]
    assert security.findings == "security issue in app/main.py"
    assert performance.findings == "performance issue in app/main.py"
    assert result["final_review"] is not None


# ---------------------------------------------------------------------------
# 4. Splitting a large PR into batches that fit the per-minute token limit
# ---------------------------------------------------------------------------


def _patch_with_lines(count: int, start: int = 1) -> str:
    body = "\n".join(f"+line_{i} = compute_something_long({i})" for i in range(start, start + count))
    return f"@@ -0,0 +{start},{count} @@\n{body}"


def test_plan_batches_keeps_small_pr_in_one_batch():
    patches = {"a.py": _patch_with_lines(5), "b.py": _patch_with_lines(5)}
    batches = _plan_batches(["a.py", "b.py"], [], patches, budget_tokens=1500)
    assert len(batches) == 1
    assert [block.splitlines()[0] for block in batches[0][0]] == ["### a.py", "### b.py"]


def test_plan_batches_splits_across_files_and_within_an_oversized_diff():
    patches = {"big.py": _patch_with_lines(400), "small.py": _patch_with_lines(5)}
    batches = _plan_batches(["big.py", "small.py"], ["huge.bin"], patches, budget_tokens=1500)
    diff_blocks = [block for blocks, _ in batches for block in blocks]
    parts = [block for block in diff_blocks if block.startswith("### big.py (part ")]
    assert len(parts) > 1
    # Every line of the big diff appears exactly once, with its real line number.
    rendered_numbers = [
        line.split("|")[0].strip() for block in parts for line in block.splitlines()[1:] if "|" in line
    ]
    assert rendered_numbers == [str(i) for i in range(1, 401)]
    # A non-diffable file always gets a batch of its own.
    assert batches[-1] == ([], ["huge.bin"])


def test_merge_batch_outcomes_combines_reports_and_flags_missing_batches():
    fix_a = InlineSuggestion(file_path="a.py", line=1, suggested_code="x", comment="a")
    fix_b = InlineSuggestion(file_path="b.py", line=2, suggested_code="y", comment="b")
    outcomes = [
        {
            "findings": SpecialistFindings(findings="Bug in a.py", validation_notes="Suite passes.", suggested_fixes=[fix_a]),
            "fetched_files": ["a.py"],
            "circuit_broken": False,
            "last_message": "1",
        },
        {
            "findings": SpecialistFindings(findings="None found.", validation_notes="Suite passes.", suggested_fixes=[]),
            "fetched_files": ["c.py"],
            "circuit_broken": False,
            "last_message": "2",
        },
        {"findings": None, "fetched_files": [], "circuit_broken": False, "last_message": "3"},
        {
            "findings": SpecialistFindings(findings="Leak in b.py", validation_notes="None found.", suggested_fixes=[fix_b]),
            "fetched_files": ["b.py"],
            "circuit_broken": False,
            "last_message": "4",
        },
    ]
    merged = _merge_batch_outcomes(outcomes)
    findings = merged["findings"]
    assert findings.findings.startswith("Bug in a.py\n\nLeak in b.py")
    assert "batch(es) 3 of 4 produced no report" in findings.findings
    assert findings.validation_notes == "Suite passes."
    assert findings.suggested_fixes == [fix_a, fix_b]
    assert merged["fetched_files"] == ["a.py", "b.py", "c.py"]
    assert merged["last_message"] == "4"


def test_fetch_file_contents_reads_a_line_range_with_real_numbers(tmp_path):
    (tmp_path / "mod.py").write_text("\n".join(f"line {i}" for i in range(1, 21)))
    out = fetch_file_contents(str(tmp_path), "mod.py", start_line=5, end_line=7)
    # Width still padded to the whole file's line count, same as a full read.
    assert out.splitlines() == [" 5 | line 5", " 6 | line 6", " 7 | line 7", "[lines 5-7 of 20]"]
    assert fetch_file_contents(str(tmp_path), "mod.py", start_line=30).startswith("Error:")
    full = fetch_file_contents(str(tmp_path), "mod.py")
    assert full.splitlines()[0] == " 1 | line 1" and "[lines" not in full
