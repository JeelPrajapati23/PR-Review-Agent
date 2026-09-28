import logging
import re
import sys
from pathlib import Path
from typing import NotRequired, TypedDict

from groq import APIConnectionError, APITimeoutError, RateLimitError
from langchain_core.messages import AIMessage
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.redis import AsyncRedisSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.config import get_settings
from app.github_client import (
    GitHubNotifyError,
    format_diff_for_review,
    get_diff_commentable_lines,
    post_review,
    set_commit_status,
)
from app.telemetry import record_usage

logger = logging.getLogger(__name__)

MCP_SERVERS_DIR = Path(__file__).resolve().parent.parent / "mcp_servers"

# Stop once every tool call in a turn errors for this many turns in a row --
# signals an environment problem (bad repo_path, a tool crash) rather than
# something fixable by the agent retrying.
MAX_CONSECUTIVE_TOOL_ERROR_TURNS = 2


class ReviewAgentState(AgentState):
    consecutive_tool_error_turns: NotRequired[int]
    # Required by create_react_agent whenever response_format is set and a
    # custom state_schema is provided -- it doesn't fall back to its own
    # AgentStateWithStructuredResponse in that case.
    structured_response: NotRequired[dict | BaseModel]
    # Files the PR actually modified/added, per the webhook payload. Carried
    # in state (in addition to being spelled out in the task message) so
    # other nodes/hooks can inspect it programmatically without parsing text.
    target_files: NotRequired[list[str]]

# Excluded entirely rather than just discouraged via the prompt:
# list_repo_files enables whole-repo scanning, which defeats the
# target_files/scan_local_dependencies minimal-context workflow below.
_FORBIDDEN_TOOL_NAMES = {"list_repo_files"}

# Specialists are read-only reviewers: no apply_code_patch. Concrete fixes
# are reported as suggested_fixes text instead; the Synthesizer reconciles
# each specialist's advice, not their edits.
_SPECIALIST_TOOL_NAMES = {"fetch_file_contents", "scan_local_dependencies", "run_validation_suite"}


def _trailing_tool_messages(messages: list) -> list:
    trailing = []
    for message in reversed(messages):
        if getattr(message, "type", None) != "tool":
            break
        trailing.append(message)
    return trailing


def _stop_on_repeated_tool_errors(state: ReviewAgentState) -> dict | Command:
    """pre_model_hook: runs right before each model call, i.e. right after any
    tool results from the previous turn have landed in state. Counts turns
    where every tool call errored and ends the graph early once the streak
    hits MAX_CONSECUTIVE_TOOL_ERROR_TURNS, instead of letting the agent loop
    indefinitely on a problem it can't fix (e.g. an invalid repo_path).
    """
    trailing = _trailing_tool_messages(list(state["messages"]))
    if not trailing:
        return {"consecutive_tool_error_turns": 0}

    all_errored = all(
        isinstance(msg.content, str) and msg.content.startswith("Error:") for msg in trailing
    )
    if not all_errored:
        return {"consecutive_tool_error_turns": 0}

    turns = state.get("consecutive_tool_error_turns", 0) + 1
    if turns < MAX_CONSECUTIVE_TOOL_ERROR_TURNS:
        return {"consecutive_tool_error_turns": turns}

    last_errors = "\n".join(msg.content for msg in reversed(trailing))
    logger.warning("Stopping review agent after %s consecutive tool-error turns", turns)
    return Command(
        goto=END,
        update={
            "consecutive_tool_error_turns": turns,
            "messages": [
                AIMessage(
                    content=(
                        "Stopping early: MCP tool calls returned errors for "
                        f"{turns} turns in a row, so this is likely an "
                        "environment/setup problem rather than something fixable "
                        f"by rewriting code. Last tool error(s):\n{last_errors}"
                    )
                )
            ],
        },
    )


# Shared tool-use discipline for every specialist persona below: identical
# READ FIRST / grounding rules to the previous single-agent SYSTEM_PROMPT,
# minus the WRITE/SELF-HEAL steps -- specialists have no apply_code_patch.
_PANEL_TOOL_DISCIPLINE = """You do not have a repo-wide file listing tool, and must not try to
reconstruct one -- scanning the whole repository wastes context and is forbidden. The task message
gives you the PR's actual diffs directly, under a "DIFFS" section -- one per target file that
GitHub could compute a diff for, already showing only the changed hunks (plus a little surrounding
context) rather than the whole file. Review those diffs directly; do NOT call fetch_file_contents
on a file whose diff is already shown just to re-read the same file in full -- Groq's per-minute
token budget is tight, and re-fetching a file you already have is the single biggest way to blow
through it. Only call fetch_file_contents on a diff-shown file as an exception, when a hunk
references something not visible in the shown lines (a helper function, a class field, an import)
and you genuinely need to see it to confirm or refute a finding -- not as a default step. If the
task message also lists a "FILES REQUIRING FULL FETCH" section (files too large or binary for
GitHub to diff), you must call fetch_file_contents on each of those before reviewing them, same as
before. If a file (from either section) has a local (same-project) import whose logic matters to
your review, call scan_local_dependencies on that file to resolve just that dependency's path, then
fetch_file_contents on the paths it returns. Never guess at file contents or a file's existence.

You are a read-only reviewer on this panel -- you have no patch tool. When you find a concrete,
line-level fix, report it as a structured suggested fix (file, line, replacement code, one-sentence
reason) in addition to describing it in your findings text. The replacement code replaces exactly
the targeted line(s) -- if it rewrites several consecutive lines, also give end_line (the last line
it replaces), or the untouched originals will remain below it as duplicates. Every line shown to you -- whether in a
DIFFS section or returned by fetch_file_contents -- is prefixed with its exact 1-indexed new-file
line number (format: "N | code"; a removed line in a diff has no number, since it doesn't exist in
the new file and can't be a fix target) specifically so you can copy that number directly rather
than counting lines yourself -- always use the number exactly as given, never estimate or
recompute it, especially when a file has multiple similar-looking blocks. The "N | " prefix is not
part of the source: never include it in a suggested fix's replacement code. Only report a suggested
fix for a file you actually saw (via a DIFFS section or a fetch_file_contents call) and a line
number that source actually showed you. run_validation_suite runs the PR's existing test suite; you
cannot write patches to fix a failure it surfaces. When you are required to call it is spelled out
below, since "call it if relevant" on its own is too easy to skip once you already have enough for
a finding.

Always use the exact repo_path given to you in the task message when calling tools; never ask the
user for one. You are a static reviewer, not an interpreter: reason only about the code as written,
never simulate, predict, or narrate what it would print if run.

Before asserting that specific code triggers a specific runtime behavior (e.g. "this raises
IndexError" or "this could throw KeyError"), verify that claim actually follows from the language
and library semantics as reflected in the code you read. An incorrect claim about what the code
does is worse than missing the real issue, since it erodes trust in every other finding in this
review. If you are not confident a suspected issue is real, do not report it as a substitute for
not having found anything else in your scope -- write "None found." for that field instead of a
plausible-sounding but unverified claim.

FINAL STEP: Once you are done, stop making tool calls. A structured findings summary will then be
generated from this conversation, so make sure every issue you report was actually read or
observed via a tool call first, rather than assumed or simulated."""


def _persona_prompt(role: str, focus: str, validation_guidance: str) -> str:
    return (
        f"You are {role} on a multi-specialist code review panel reviewing a pull request via "
        f"MCP tools. Your job is scoped STRICTLY to: {focus} Do not comment on issues outside this "
        "scope -- another specialist on the panel covers those, and a Synthesizer will merge both "
        "of your reports afterwards.\n\n"
        f"{_PANEL_TOOL_DISCIPLINE}\n\n"
        f"VALIDATION: {validation_guidance}"
    )


SECURITY_WARDEN_PROMPT = _persona_prompt(
    "the Security Warden",
    "identifying data leaks (secrets/PII/tokens in code, logs, or output), dependency "
    "vulnerabilities, OWASP Top 10 flaws (injection, broken auth, unsafe deserialization, path "
    "traversal, SSRF, etc.), and code-injection risks.",
    "Call run_validation_suite if any finding you are about to report asserts or depends on the "
    "test suite's current behavior -- e.g. before claiming a vulnerable path is untested, or before "
    "reporting a finding that contradicts what a passing suite would suggest, confirm the suite's "
    "actual current result rather than assuming it. This panel exists specifically to catch issues "
    "that slip past a passing test suite, so if the suite passes despite a real vulnerability you "
    "found, say so explicitly in validation_notes -- that gap is itself worth surfacing. If none of "
    "your findings depend on the suite's state, skip the call and write 'Not run: no findings "
    "depended on test suite state.' in validation_notes -- do not write 'None found.' for that case.",
)

PERFORMANCE_ARCHITECT_PROMPT = _persona_prompt(
    "the Performance & Logic Architect",
    "algorithmic correctness, edge cases the code fails to handle, Big-O time/space efficiency, "
    "and optimal/control-flow soundness -- including SILENT failure modes as much as ones that "
    "throw an exception. Specifically look for: off-by-one slicing/indexing that quietly drops or "
    "corrupts data instead of erroring; unsynchronized read-modify-write access to shared state "
    "from concurrent callers (race conditions/check-then-act bugs); and caches or memoized results "
    "keyed on the wrong (or incomplete) set of inputs, returning stale results for new inputs. A "
    "bug that never crashes but silently returns the wrong answer is just as real as one that "
    "raises, and is often more dangerous because nothing alerts anyone to it.",
    "Call run_validation_suite at least once before finishing this review, unless target_files "
    "contains nothing but non-executable changes (docs, config, markdown). This panel exists "
    "specifically to catch issues a passing test suite already missed, so for each correctness/"
    "logic issue you report, note in validation_notes whether the existing suite actually exercises "
    "the code path behind it -- 'suite passes but does not cover the affected branch/input' is a "
    "materially more useful observation than just 'suite passes.' Only write 'Not run: target_files "
    "contained no executable logic.' in validation_notes if you truly skipped the call for that "
    "reason -- do not write 'None found.' for that case.",
)

# Used for each specialist's structured-output extraction pass, not the
# tool-using ReAct loop above: langgraph does not persist the ReAct loop's
# injected `prompt=` into state["messages"], so this pass needs its own
# self-contained instructions.
SPECIALIST_STRUCTURED_PROMPT = """Using only the tool calls and tool results already in this
conversation, produce your structured findings for this specialist review. Every field must be
grounded in the actual file contents returned by fetch_file_contents calls above -- describe real
code you read. Never invent, simulate, or describe code that was not actually fetched in this
conversation. Every issue you report in findings MUST explicitly name the exact file it applies to
(its repo-relative path, e.g. "in app/main.py: ...") -- a downstream check verifies each issue
against the files actually fetched, so never describe an issue without naming its file. For every
issue that has a concrete, line-level fix, also add an entry to suggested_fixes -- do not rely on
mentioning the fix in prose alone. Write "None found." for the findings field if there is nothing
to report. For validation_notes specifically: write "None found." only if you called
run_validation_suite and it surfaced nothing relevant; if your persona's VALIDATION guidance
required the call and you skipped it for a stated reason, write "Not run: <reason>." instead."""


class InlineSuggestion(BaseModel):
    """A single line-level fix, rendered as a GitHub suggestion-block review comment."""

    file_path: str = Field(description="Repo-relative path of the file this suggestion applies to.")
    line: int = Field(description="1-indexed line number in the file's current content where the fix applies.")
    end_line: int | None = Field(
        default=None,
        description="Last line (inclusive) this fix replaces, when it rewrites several consecutive lines "
        "starting at `line`. Omit for a single-line fix. suggested_code replaces lines line..end_line "
        "entirely, so if your replacement is a rewrite of the next few lines too, you MUST cover them "
        "here -- otherwise those original lines stay in place below your replacement and get duplicated.",
    )
    suggested_code: str = Field(
        description="Exact replacement text for that line/block, with no diff markers or line numbers. "
        "Must keep the original line's full leading indentation (e.g. the 4 spaces before a statement "
        "inside a function body) -- GitHub replaces the whole line verbatim, so stripped indentation "
        "produces broken code."
    )
    comment: str = Field(description="One-sentence explanation of the issue being fixed.")


class SpecialistFindings(BaseModel):
    """Structured findings from a single specialist persona on the review panel."""

    findings: str = Field(
        description="Concrete issues found within this persona's scope, grounded in the code "
        "actually read. 'None found.' if nothing to report."
    )
    validation_notes: str = Field(
        description="Relevant run_validation_suite output/observations. If you were required to "
        "call it and skipped for a stated reason (per your persona's VALIDATION guidance), write "
        "'Not run: <reason>.' here -- never 'None found.' for a skipped-but-required call. 'None "
        "found.' is only for 'I called it and it surfaced nothing relevant.'"
    )
    suggested_fixes: list[InlineSuggestion] = Field(
        default_factory=list,
        description="Concrete line-level fixes for issues in your findings, one per fixable issue. "
        "Only include a fix for a file/line you actually fetched via fetch_file_contents.",
    )


SYNTHESIZER_PROMPT = """You are the Synthesizer Supervisor on a code review panel. You are given two
specialist reports on the same pull request: one from the Security Warden (data leaks, dependency
vulnerabilities, OWASP flaws, code-injection risks) and one from the Performance & Logic Architect
(algorithmic correctness, edge cases, Big-O efficiency, control flow). Combine them into a single,
coherent review:
- Deduplicate overlapping points raised by both specialists into one clear statement.
- If the two specialists give conflicting advice about the same code, resolve it by favoring
  correctness and security over raw performance, and briefly explain the tradeoff.
- Route each point into the correct section: security/vulnerability findings into
  security_correctness_issues, algorithmic/Big-O findings into performance_observations, unhandled
  edge cases into edge_cases, and any structural/control-flow feedback into
  architectural_suggestions.
- Route by what an issue actually IS, not by which specialist reported it or how they phrased it.
  A specialist will sometimes describe a structural issue using language that sounds like their own
  primary focus -- e.g. the Performance & Logic Architect calling duplicated code "redundant
  calculations that can be simplified." Duplicated/copy-pasted logic, dead or commented-out code,
  unclear naming, magic numbers, deep nesting, and inconsistent patterns (mixed logging styles,
  inconsistent error handling) are ALWAYS architectural_suggestions, even when the reporting
  specialist's own wording sounds performance- or correctness-flavored. Reserve
  performance_observations for genuine runtime cost: algorithmic complexity/Big-O, unnecessary
  recomputation, or inefficient data structures -- not code that is merely duplicated or hard to
  read.
- Every issue mentioned in summary must also appear in one of the five detailed sections below.
  Never describe a finding only in summary while every section says "None found." for it -- if a
  finding is worth summarizing, it is worth routing to its section (most often
  architectural_suggestions for structural/maintainability findings).
- Combine both specialists' validation_notes into validation_outcome. Preserve each one's own
  wording on whether the suite was actually run -- if either specialist wrote 'Not run: <reason>.',
  carry that reason into validation_outcome verbatim rather than collapsing it into "None found.";
  "None found." in validation_outcome must mean the suite was run and surfaced nothing relevant, not
  that it was skipped.
- CRITICAL: every specialist report names the exact file each issue applies to. You MUST preserve
  that exact file name in your own wording for every issue you carry into the output -- a
  downstream check verifies each section against the files actually reviewed, so a section that
  paraphrases an issue away from its file name will be discarded even if correct.
- Never invent an issue neither specialist actually reported, and never describe code neither
  report mentions.
- Write "None found." for any of the other four sections neither specialist populated."""


class ReviewOutput(BaseModel):
    """Structured PR review, synthesized from the review panel's specialist findings."""

    summary: str = Field(
        description="1-3 sentence summary of what was reviewed. 'None found.' if nothing was reviewed."
    )
    performance_observations: str = Field(
        description="Genuine runtime-cost issues in the code actually read: algorithmic complexity/"
        "Big-O, unnecessary recomputation, or inefficient data structures. Does NOT include code "
        "duplication, dead code, naming, or other structural/maintainability issues -- those belong "
        "in architectural_suggestions even if worded in performance-sounding terms. 'None found.' if "
        "nothing to report."
    )
    edge_cases: str = Field(
        description="Edge cases the reviewed code does not handle. 'None found.' if nothing to report."
    )
    security_correctness_issues: str = Field(
        description="Security vulnerabilities or correctness bugs found in the code actually read. "
        "'None found.' if nothing to report."
    )
    architectural_suggestions: str = Field(
        description="Structural/maintainability issues in the reviewed code: duplicated/copy-pasted "
        "logic, dead or commented-out code, unclear naming, magic numbers, deep nesting, and "
        "inconsistent patterns (e.g. mixed logging styles, inconsistent error handling). These belong "
        "here even if a specialist described them in performance- or correctness-sounding language. "
        "'None found.' if nothing to report."
    )
    validation_outcome: str = Field(
        description="Result of run_validation_suite calls made during this review. 'None found.' means "
        "the suite was run and nothing relevant surfaced. If either specialist skipped the required "
        "call, state that as 'Not run: <reason>.' instead -- do not write 'None found.' for a skipped "
        "call."
    )
    inline_suggestions: list[InlineSuggestion] = Field(
        default_factory=list,
        description="Not populated by this model call -- overwritten by synthesizer_node from the "
        "specialists' own suggested_fixes after this structured call returns.",
    )


class PanelState(TypedDict):
    """Shared state for the review panel's supervisor graph.

    security_warden and performance_architect each write to their own,
    disjoint keys, never a shared 'messages' channel -- required for
    LangGraph to allow two nodes to update state in the same superstep
    without an InvalidUpdateError.
    """

    task_message: str
    target_files: list[str]
    diff_shown_files: list[str]
    security_findings: NotRequired[SpecialistFindings | None]
    security_fetched_files: NotRequired[list[str]]
    security_circuit_broken: NotRequired[bool]
    security_last_message: NotRequired[str]
    performance_findings: NotRequired[SpecialistFindings | None]
    performance_fetched_files: NotRequired[list[str]]
    performance_circuit_broken: NotRequired[bool]
    performance_last_message: NotRequired[str]
    final_review: NotRequired[ReviewOutput | None]


def _notify(repository: str, sha: str, state: str, description: str, pr_number: int) -> None:
    try:
        set_commit_status(repository, sha, state, description)
    except GitHubNotifyError:
        logger.exception("Failed to set '%s' commit status for %s#%s", state, repository, pr_number)


def _post_review(
    repository: str, pr_number: int, sha: str, body: str, comments: list[dict] | None = None
) -> None:
    try:
        post_review(repository, pr_number, sha, body, comments=comments or None)
    except GitHubNotifyError:
        logger.exception("Failed to post review for %s#%s", repository, pr_number)


def _tool_message_text(message) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)


def _sum_usage_metadata(messages: list) -> tuple[int, int]:
    """Total (prompt_tokens, completion_tokens) across every AIMessage's
    usage_metadata in a conversation. A ReAct loop makes one model call per
    turn, each producing its own AIMessage with its own usage figures, so
    summing across all of them gives this invocation's total token spend.

    Note: on a resumed checkpointed thread, messages include prior turns
    already recorded by an earlier attempt, so a retry's totals overlap with
    what was already written to Redis.
    """
    prompt_tokens = 0
    completion_tokens = 0
    for message in messages:
        usage = getattr(message, "usage_metadata", None)
        if usage:
            prompt_tokens += usage.get("input_tokens", 0)
            completion_tokens += usage.get("output_tokens", 0)
    return prompt_tokens, completion_tokens


def _fetched_file_names(messages: list) -> set[str]:
    """Basenames of files fetch_file_contents actually returned content for
    (as opposed to erroring) during this run, used to sanity-check that the
    final review is grounded in files the agent really read.
    """
    call_id_to_name = {
        call["id"]: Path(call["args"]["file_path"]).name.lower()
        for message in messages
        for call in getattr(message, "tool_calls", None) or []
        if call.get("name") == "fetch_file_contents" and call.get("args", {}).get("file_path")
    }

    fetched = set()
    for message in messages:
        if getattr(message, "type", None) != "tool":
            continue
        name = call_id_to_name.get(getattr(message, "tool_call_id", None))
        if name and not _tool_message_text(message).startswith("Error:"):
            fetched.add(name)
    return fetched


def _combine_fetched_files(messages: list, diff_shown_files: list[str]) -> set[str]:
    """Files this specialist should be treated as having actually reviewed:
    those fetched via fetch_file_contents, plus any whose diff was shown
    directly in the task message. A diff-shown file is GitHub-verified
    content, not model output, so counting it here gives the same grounding
    guarantee an actual fetch_file_contents call would -- just via a cheaper
    delivery path (see _PANEL_TOOL_DISCIPLINE).
    """
    return _fetched_file_names(messages) | set(diff_shown_files)


def _format_review(review: ReviewOutput) -> str:
    return (
        f"## Summary\n{review.summary}\n\n"
        f"## Performance Observations\n{review.performance_observations}\n\n"
        f"## Edge Cases\n{review.edge_cases}\n\n"
        f"## Security & Correctness Issues\n{review.security_correctness_issues}\n\n"
        f"## Architectural Suggestions\n{review.architectural_suggestions}\n\n"
        f"## Validation Outcome\n{review.validation_outcome}"
    )


def _leading_whitespace(text: str) -> str:
    return text[: len(text) - len(text.lstrip())]


def _restore_indentation(suggested_code: str, original_line: str) -> str:
    """Re-indent a suggestion whose first line lost the original line's
    leading whitespace. The model routinely strips it (seen live: a fix for an
    indented `query = ...` line inside a function came back at column 0, which
    GitHub's "Commit suggestion" would have applied as an IndentationError).

    Two observed shapes: only the first line stripped (later lines still at or
    past the original indent -- fix just the first line), or the whole block
    dedented uniformly (pad every non-blank line by the missing amount). A
    first line that opens a block (ends in ":" or "{") whose body sits at
    exactly the original indent can only be the uniform-dedent case, since
    that body must be nested deeper than its opener.
    """
    lines = suggested_code.split("\n")
    first = next((i for i, line in enumerate(lines) if line.strip()), None)
    if first is None:
        return suggested_code

    original_indent = _leading_whitespace(original_line)
    first_indent = _leading_whitespace(lines[first])
    if len(first_indent) >= len(original_indent):
        return suggested_code

    rest = [line for line in lines[first + 1 :] if line.strip()]
    rest_indent = min((len(_leading_whitespace(line)) for line in rest), default=0)
    opens_block = lines[first].rstrip().endswith((":", "{"))
    if rest and rest_indent >= len(original_indent) and not (opens_block and rest_indent == len(original_indent)):
        lines[first] = original_indent + lines[first].lstrip()
        return "\n".join(lines)

    pad = original_indent[len(first_indent) :]
    return "\n".join(pad + line if line.strip() else line for line in lines)


def _format_suggestion_body(suggestion: InlineSuggestion) -> str:
    return f"{suggestion.comment}\n\n```suggestion\n{suggestion.suggested_code}\n```"


def _first_token(line: str) -> str | None:
    match = re.match(r"\s*(\w+|\S)", line)
    return match.group(1) if match else None


def _replaced_line_range(suggestion: InlineSuggestion, file_lines: dict[int, str]) -> tuple[int, int]:
    """(start, end) of the original lines a suggestion's replacement covers.

    A GitHub suggestion replaces exactly the commented lines, so a multi-line
    replacement anchored to a single line keeps the originals below it --
    seen live: a fix rewriting both `query = ...` and the `return ...` after
    it was anchored to the query line alone, so committing it would have left
    two returns. When the model under-reports end_line, each extra
    replacement line that starts with the same token as the next original
    line (e.g. both `return`) is treated as a rewrite of it and pulled into
    the range; anything else stays a genuine insertion.
    """
    start = suggestion.line
    end = max(suggestion.end_line or start, start)
    replacement = suggestion.suggested_code.split("\n")
    for extra in replacement[end - start + 1 :]:
        following = file_lines.get(end + 1)
        if following is None or _first_token(extra) is None or _first_token(extra) != _first_token(following):
            break
        end += 1
    return start, end


def _build_inline_comments(
    suggestions: list[InlineSuggestion],
    fetched_files: set[str],
    commentable_lines: dict[str, dict[int, str]],
) -> list[dict]:
    """Convert InlineSuggestions into GitHub inline review-comment dicts.

    Drops any suggestion whose file was not actually fetched during this run
    (same fabrication guard as _is_grounded), and separately drops any
    suggestion whose replaced lines aren't all part of the PR's actual diff --
    GitHub's create_review rejects the entire review if any comment can't be
    resolved against the diff. Surviving suggestions get the target line's
    original indentation restored (see _restore_indentation), and span every
    line they replace (see _replaced_line_range).

    Both specialists often propose the same fix for the same lines (seen
    live: two identical suggestion comments on one PR), and two suggestions
    over overlapping lines can't both be applied anyway -- so any suggestion
    overlapping one already kept for that file is dropped. suggestions lists
    the Security Warden's fixes first, so this favors security on conflict,
    same as the Synthesizer's own tie-break.
    """
    comments = []
    kept_ranges: dict[str, list[tuple[int, int]]] = {}
    for suggestion in suggestions:
        basename = Path(suggestion.file_path).name.lower()
        if basename not in fetched_files:
            continue

        resolved_path = suggestion.file_path
        if resolved_path not in commentable_lines:
            resolved_path = next(
                (path for path in commentable_lines if Path(path).name.lower() == basename), None
            )
        if resolved_path is None:
            continue
        file_lines = commentable_lines[resolved_path]
        if suggestion.line not in file_lines:
            continue

        suggestion = suggestion.model_copy(
            update={"suggested_code": _restore_indentation(suggestion.suggested_code, file_lines[suggestion.line])}
        )
        start, end = _replaced_line_range(suggestion, file_lines)
        # A range crossing a gap between hunks isn't commentable as one block.
        if any(n not in file_lines for n in range(start, end + 1)):
            continue
        ranges = kept_ranges.setdefault(resolved_path, [])
        if any(start <= kept_end and kept_start <= end for kept_start, kept_end in ranges):
            continue
        ranges.append((start, end))

        comment = {"path": resolved_path, "line": end, "side": "RIGHT", "body": _format_suggestion_body(suggestion)}
        if end > start:
            comment.update(start_line=start, start_side="RIGHT")
        comments.append(comment)
    return comments


def _is_grounded(formatted_review: str, fetched_files: set[str]) -> bool:
    """Heuristic grounding check: a genuine review should reference at least
    one file the agent actually fetched, guarding against a fabricated
    review not grounded in real tool results.
    """
    if not fetched_files:
        return False
    lowered = formatted_review.lower()
    return any(name in lowered for name in fetched_files)


def _thread_id_for(repository: str, pr_number: int, sha: str) -> str:
    """Redis checkpoint thread_id for a single code snapshot's review.

    Scoped to the head sha (not just repo+PR), so a later 'synchronize' event
    (new commits on the same PR) gets its own clean thread instead of
    resuming the previous commit's tool-call history. A retried task for the
    same sha still resumes its own thread. Each specialist further
    namespaces its own sub-agent thread off of this base id.
    """
    return f"pr:{repository}:{pr_number}:{sha}"


def _build_mcp_client() -> MultiServerMCPClient:
    return MultiServerMCPClient(
        {
            "code_server": {
                "command": sys.executable,
                "args": [str(MCP_SERVERS_DIR / "code_server.py")],
                "transport": "stdio",
            },
            "tester_server": {
                "command": sys.executable,
                "args": [str(MCP_SERVERS_DIR / "tester_server.py")],
                "transport": "stdio",
            },
        }
    )


async def _run_specialist(
    persona_key: str,
    prompt: str,
    tools: list,
    checkpointer: AsyncRedisSaver,
    base_thread_id: str,
    task_message: str,
    target_files: list[str],
    diff_shown_files: list[str],
) -> dict:
    """Run one specialist persona's own ReAct sub-agent to completion.

    Each specialist is its own compiled graph with its own Redis checkpoint
    thread (base_thread_id suffixed with the persona key), so its internal
    tool-call history is durable independently of the panel's own state.
    """
    specialist_tools = [tool for tool in tools if tool.name in _SPECIALIST_TOOL_NAMES]
    settings = get_settings()
    model = ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0.1,
        max_tokens=4000,
        max_retries=2,
    )
    agent = create_react_agent(
        model,
        specialist_tools,
        prompt=prompt,
        response_format=(SPECIALIST_STRUCTURED_PROMPT, SpecialistFindings),
        state_schema=ReviewAgentState,
        pre_model_hook=_stop_on_repeated_tool_errors,
        checkpointer=checkpointer,
    )
    result = await agent.ainvoke(
        {"messages": [("user", task_message)], "target_files": target_files},
        config={
            "recursion_limit": 25,
            "configurable": {"thread_id": f"{base_thread_id}:{persona_key}"},
        },
    )
    messages = result["messages"]
    prompt_tokens, completion_tokens = _sum_usage_metadata(messages)
    await record_usage(settings.groq_model, prompt_tokens, completion_tokens)
    fetched_files = _combine_fetched_files(messages, diff_shown_files)
    return {
        "findings": result.get("structured_response"),
        "fetched_files": sorted(fetched_files),
        "circuit_broken": result.get("consecutive_tool_error_turns", 0) >= MAX_CONSECUTIVE_TOOL_ERROR_TURNS,
        "last_message": messages[-1].content if messages else "",
    }


def _findings_block(label: str, findings: SpecialistFindings | None) -> str:
    if findings is None:
        return f"## {label}\n(No report produced -- this specialist did not complete.)"
    return f"## {label}\nFindings: {findings.findings}\nValidation notes: {findings.validation_notes}"


async def _synthesize(security: SpecialistFindings | None, performance: SpecialistFindings | None) -> ReviewOutput:
    """Non-tool-using LLM pass: merges both specialists' reports into the
    final ReviewOutput. No tool access is needed or given here -- it only
    reconciles text the specialists already produced.
    """
    settings = get_settings()
    model = ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0.1,
        max_tokens=4000,
        max_retries=2,
    )
    panel_report = (
        _findings_block("Security Warden Report", security)
        + "\n\n"
        + _findings_block("Performance & Logic Architect Report", performance)
    )
    # include_raw=True returns {"raw", "parsed", "parsing_error"} instead of
    # just the parsed object -- needed to reach the raw AIMessage's
    # usage_metadata, which ReviewOutput alone doesn't carry. A parsing
    # failure surfaces as parsed=None, handled by the None-check downstream.
    response = await model.with_structured_output(ReviewOutput, include_raw=True).ainvoke(
        [("system", SYNTHESIZER_PROMPT), ("user", panel_report)]
    )
    raw_message = response.get("raw")
    usage = getattr(raw_message, "usage_metadata", None) or {}
    await record_usage(settings.groq_model, usage.get("input_tokens", 0), usage.get("output_tokens", 0))
    return response.get("parsed")


def _build_panel_graph(tools: list, checkpointer: AsyncRedisSaver, base_thread_id: str):
    """Sequential supervisor graph: security_warden runs, then
    performance_architect, then the Synthesizer. The two specialists don't
    depend on each other's output -- this ordering exists purely to keep
    peak per-minute token usage against the Groq API to one specialist call
    at a time instead of two concurrent ones, since the free-tier TPM limit
    can't absorb both firing at once.
    """

    async def security_node(state: PanelState) -> dict:
        outcome = await _run_specialist(
            "security",
            SECURITY_WARDEN_PROMPT,
            tools,
            checkpointer,
            base_thread_id,
            state["task_message"],
            state["target_files"],
            state["diff_shown_files"],
        )
        return {
            "security_findings": outcome["findings"],
            "security_fetched_files": outcome["fetched_files"],
            "security_circuit_broken": outcome["circuit_broken"],
            "security_last_message": outcome["last_message"],
        }

    async def performance_node(state: PanelState) -> dict:
        outcome = await _run_specialist(
            "performance",
            PERFORMANCE_ARCHITECT_PROMPT,
            tools,
            checkpointer,
            base_thread_id,
            state["task_message"],
            state["target_files"],
            state["diff_shown_files"],
        )
        return {
            "performance_findings": outcome["findings"],
            "performance_fetched_files": outcome["fetched_files"],
            "performance_circuit_broken": outcome["circuit_broken"],
            "performance_last_message": outcome["last_message"],
        }

    async def synthesizer_node(state: PanelState) -> dict:
        security = state.get("security_findings")
        performance = state.get("performance_findings")
        review = await _synthesize(security, performance)
        if review is not None:
            # Built from each specialist's own suggested_fixes directly,
            # not extracted by the Synthesizer LLM from free text.
            review.inline_suggestions = (security.suggested_fixes if security else []) + (
                performance.suggested_fixes if performance else []
            )
        return {"final_review": review}

    graph = StateGraph(PanelState)
    graph.add_node("security_warden", security_node)
    graph.add_node("performance_architect", performance_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_edge(START, "security_warden")
    graph.add_edge("security_warden", "performance_architect")
    graph.add_edge("performance_architect", "synthesizer")
    graph.add_edge("synthesizer", END)
    return graph.compile(checkpointer=checkpointer)


def _build_task_message(
    pr_number: int,
    repository: str,
    pull_request: dict,
    repo_path: Path,
    target_files: list[str],
    file_patches: dict[str, str],
) -> tuple[str, list[str]]:
    """Build the specialists' initial task message and the list of target-file
    basenames whose diff was embedded directly in it (diff_shown_files).

    target_files not present in file_patches (GitHub couldn't diff them --
    too large or binary) are listed separately as files to fetch in full via
    fetch_file_contents.
    """
    diffed_files = [f for f in target_files if file_patches.get(f)]
    full_fetch_files = [f for f in target_files if not file_patches.get(f)]

    sections = [
        f"Review PR #{pr_number} in repository {repository}.",
        f"Title: {pull_request['title']}",
        f"Branch: {pull_request['head']['ref']}",
        f"repo_path to use for all tool calls: {repo_path}",
    ]

    if diffed_files:
        diff_blocks = "\n\n".join(
            f"### {file_path}\n{format_diff_for_review(file_patches[file_path])}" for file_path in diffed_files
        )
        sections.append(
            "DIFFS (shown below -- review directly; call fetch_file_contents on the same file "
            "only if you need to see code outside these hunks to judge correctness):\n" + diff_blocks
        )

    if full_fetch_files:
        sections.append(
            "FILES REQUIRING FULL FETCH (no diff available -- call fetch_file_contents on each "
            f"before reviewing): {full_fetch_files}"
        )

    sections.append("Begin your review now.")
    task_message = "\n\n".join(sections)

    diff_shown_files = sorted({Path(f).name.lower() for f in diffed_files})
    return task_message, diff_shown_files


async def run_pr_review_agent(pr_metadata: dict, repo_path: Path) -> dict:
    repository = pr_metadata["repository"]["full_name"]
    pull_request = pr_metadata["pull_request"]
    pr_number = pull_request["number"]
    sha = pull_request["head"]["sha"]

    try:
        client = _build_mcp_client()
        tools = [tool for tool in await client.get_tools() if tool.name not in _FORBIDDEN_TOOL_NAMES]

        target_files = list(dict.fromkeys(
            pull_request.get("modified_files", []) + pull_request.get("added_files", [])
        ))
        file_patches = pull_request.get("file_patches", {})

        task_message, diff_shown_files = _build_task_message(
            pr_number, repository, pull_request, repo_path, target_files, file_patches
        )

        base_thread_id = _thread_id_for(repository, pr_number, sha)

        # Redis-backed checkpointing: this PR's review panel state survives a
        # worker crash/restart. The same checkpointer connection backs the
        # panel graph itself (thread_id=base_thread_id) and each specialist's
        # own sub-agent (thread_id=f"{base_thread_id}:{persona_key}").
        settings = get_settings()
        async with AsyncRedisSaver.from_conn_string(settings.redis_url) as checkpointer:
            await checkpointer.asetup()
            panel = _build_panel_graph(tools, checkpointer, base_thread_id)

            panel_result = await panel.ainvoke(
                {"task_message": task_message, "target_files": target_files, "diff_shown_files": diff_shown_files},
                config={
                    "recursion_limit": 25,
                    "configurable": {"thread_id": base_thread_id},
                },
            )

        security_broken = panel_result.get("security_circuit_broken", False)
        performance_broken = panel_result.get("performance_circuit_broken", False)
        fetched_files = set(panel_result.get("security_fetched_files", [])) | set(
            panel_result.get("performance_fetched_files", [])
        )

        if security_broken or performance_broken:
            status = "error"
            details = []
            if security_broken:
                details.append(f"Security Warden: {panel_result.get('security_last_message', '')}")
            if performance_broken:
                details.append(f"Performance & Logic Architect: {panel_result.get('performance_last_message', '')}")
            summary_text = (
                "Stopping early: repeated tool errors indicate an environment/setup problem "
                "rather than something fixable by rewriting code.\n\n" + "\n\n".join(details)
            )
            _notify(repository, sha, "failure", "Review stopped early after repeated tool errors", pr_number)
        else:
            structured = panel_result.get("final_review")
            if structured is None:
                status = "error"
                summary_text = "Review panel did not produce a structured response."
                _notify(repository, sha, "failure", "Review panel did not produce a structured response", pr_number)
            else:
                formatted_review = _format_review(structured)
                if not _is_grounded(formatted_review, fetched_files):
                    status = "error"
                    summary_text = (
                        "Review output failed grounding validation: no reviewed file was "
                        f"referenced in the analysis.\n\n{formatted_review}"
                    )
                    _notify(repository, sha, "failure", "Review output failed grounding validation", pr_number)
                else:
                    status = "completed"
                    summary_text = formatted_review
                    try:
                        commentable_lines = get_diff_commentable_lines(repository, pr_number)
                    except GitHubNotifyError:
                        logger.exception(
                            "Failed to fetch diff for inline-comment validation on %s#%s", repository, pr_number
                        )
                        commentable_lines = {}
                    inline_comments = _build_inline_comments(
                        structured.inline_suggestions, fetched_files, commentable_lines
                    )
                    _notify(repository, sha, "success", "PR review agent completed successfully", pr_number)
                    _post_review(repository, pr_number, sha, formatted_review, inline_comments)

        return {
            "repository": repository,
            "pr_number": pr_number,
            "status": status,
            "summary": summary_text,
        }
    except (RateLimitError, APIConnectionError, APITimeoutError):
        # Re-raise past this handler so Celery's autoretry_for on
        # process_pr_review_task can back off and retry, instead of this
        # transient failure being swallowed into an "error" status below.
        logger.warning(
            "Transient Groq API failure reviewing %s#%s; re-raising for Celery retry", repository, pr_number
        )
        raise
    except Exception as exc:
        logger.exception("Agent review failed for %s#%s", repository, pr_number)
        _notify(repository, sha, "failure", f"Review agent error: {exc}", pr_number)
        return {
            "repository": repository,
            "pr_number": pr_number,
            "status": "error",
            "error": str(exc),
        }
