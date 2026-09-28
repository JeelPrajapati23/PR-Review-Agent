from app.github_client import (
    _commentable_line_texts_from_patch,
    _commentable_lines_from_patch,
    format_diff_for_review,
)

# A synthetic two-hunk unified diff patch, in the same shape GitHub's PR
# Files API returns via File.patch -- used to pin down format_diff_for_review
# (and, indirectly, the shared _walk_patch_new_lines generator both it and
# _commentable_lines_from_patch are built on) without any real GitHub call.
_SAMPLE_PATCH = (
    "@@ -10,7 +10,8 @@ def foo():\n"
    " line8\n"
    " line9\n"
    "-line10_old\n"
    "+line10_new\n"
    "+line10_extra\n"
    " line11\n"
    " line12\n"
    "@@ -30,3 +31,3 @@ def bar():\n"
    " line30\n"
    "-line31_old\n"
    "+line31_new\n"
    " line32\n"
)


def test_format_diff_for_review_numbers_context_and_added_lines_by_new_file_position():
    rendered = format_diff_for_review(_SAMPLE_PATCH)

    assert "@@ -10,7 +10,8 @@ def foo():" in rendered
    assert "   10 | line8" in rendered
    assert "   11 | line9" in rendered
    assert "   12 | line10_new" in rendered
    assert "   13 | line10_extra" in rendered
    assert "   14 | line11" in rendered
    assert "   15 | line12" in rendered

    assert "@@ -30,3 +31,3 @@ def bar():" in rendered
    assert "   31 | line30" in rendered
    assert "   32 | line31_new" in rendered
    assert "   33 | line32" in rendered


def test_format_diff_for_review_marks_removed_lines_with_no_line_number():
    rendered = format_diff_for_review(_SAMPLE_PATCH)
    lines = rendered.splitlines()

    removed = [line for line in lines if "line10_old" in line or "line31_old" in line]
    assert removed == ["    - | line10_old", "    - | line31_old"]


def test_format_diff_for_review_preserves_hunk_order_and_line_order():
    rendered = format_diff_for_review(_SAMPLE_PATCH)
    lines = rendered.splitlines()

    assert lines[0] == "@@ -10,7 +10,8 @@ def foo():"
    assert lines[1] == "   10 | line8"
    first_hunk_end = next(i for i, line in enumerate(lines) if line.startswith("@@ -30"))
    assert lines[first_hunk_end] == "@@ -30,3 +31,3 @@ def bar():"


def test_commentable_lines_from_patch_excludes_removed_lines():
    # Regression-style check: format_diff_for_review and
    # _commentable_lines_from_patch must agree on which new-file line numbers
    # exist, since app/agent.py's inline-suggestion validation depends on
    # exactly that agreement (a SUGGESTION line number copied from the
    # rendered diff must always be a commentable line).
    commentable = _commentable_lines_from_patch(_SAMPLE_PATCH)

    assert commentable == {10, 11, 12, 13, 14, 15, 31, 32, 33}


def test_commentable_line_texts_keep_leading_indentation():
    # app/agent.py restores a suggestion's stripped indentation from this
    # text, so the diff's own "+"/" " marker must be removed without also
    # eating the line's real leading whitespace.
    patch = "@@ -0,0 +1,2 @@\n+def get_user(conn, username):\n+    query = 'SELECT 1'\n"

    assert _commentable_line_texts_from_patch(patch) == {
        1: "def get_user(conn, username):",
        2: "    query = 'SELECT 1'",
    }
