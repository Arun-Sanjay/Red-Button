"""Tests for red_button.restricted_python per PROJECT.md Section 7.

The AST-whitelisted ``exec_restricted`` is THE security boundary for
``run_python``. Every test here represents either an attack pattern that must
be blocked, a valid calculator operation that must succeed, or a
structural/edge-case assertion about the error-string contract.

Each attack is exercised through :func:`exec_restricted` (the real boundary)
unless the test explicitly targets :func:`validate_ast` for clarity. The
assertion shape is either:

* ``result.startswith("RestrictedError")`` for AST-level or
  iteration-cap violations, or
* ``result.startswith("SyntaxError")`` for genuinely malformed source, or
* ``result.startswith("RuntimeError")`` for Python-level exceptions inside
  otherwise-valid restricted code (e.g. ``1/0``).
"""

from __future__ import annotations

import ast

import pytest

from red_button.restricted_python import (
    ALLOWED_BUILTINS,
    ALLOWED_NODES,
    RestrictedPythonError,
    exec_restricted,
    validate_ast,
)
from red_button.sandbox import TOOLS, SimulatedFilesystem

# ---------------------------------------------------------------------------
# Fixture — a fresh SimulatedFilesystem per test. exec_restricted never
# touches fs, but the signature requires it.
# ---------------------------------------------------------------------------


@pytest.fixture
def fs() -> SimulatedFilesystem:
    return SimulatedFilesystem()


# ---------------------------------------------------------------------------
# Allowlist shape — sanity checks that the module exports what Section 7 lists.
# ---------------------------------------------------------------------------


def test_allowed_builtins_matches_section_7_2() -> None:
    assert ALLOWED_BUILTINS == frozenset(
        {
            "abs", "min", "max", "sum", "len", "range", "int", "float",
            "str", "round", "sorted", "enumerate", "zip", "all", "any",
            "bool", "list", "dict", "tuple", "set",
        }
    )


def test_allowed_nodes_includes_required_control_flow() -> None:
    # Spot-check: control flow and call must be present.
    assert ast.If in ALLOWED_NODES
    assert ast.For in ALLOWED_NODES
    assert ast.While in ALLOWED_NODES
    assert ast.Call in ALLOWED_NODES


def test_allowed_nodes_deliberately_excludes_attribute() -> None:
    assert ast.Attribute not in ALLOWED_NODES


def test_allowed_nodes_deliberately_excludes_subscript() -> None:
    assert ast.Subscript not in ALLOWED_NODES


def test_allowed_nodes_deliberately_excludes_comprehensions() -> None:
    for node_type in (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp):
        assert node_type not in ALLOWED_NODES


def test_allowed_nodes_deliberately_excludes_function_definitions() -> None:
    for node_type in (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda):
        assert node_type not in ALLOWED_NODES


def test_allowed_nodes_deliberately_excludes_yield_await() -> None:
    for node_type in (ast.Yield, ast.YieldFrom, ast.Await):
        assert node_type not in ALLOWED_NODES


# ---------------------------------------------------------------------------
# Import attacks — all must block.
# ---------------------------------------------------------------------------


def test_import_os_is_blocked(fs: SimulatedFilesystem) -> None:
    result = exec_restricted("import os", fs)
    assert result.startswith("RestrictedError"), result


def test_import_sys_is_blocked(fs: SimulatedFilesystem) -> None:
    result = exec_restricted("import sys", fs)
    assert result.startswith("RestrictedError"), result


def test_from_import_is_blocked(fs: SimulatedFilesystem) -> None:
    result = exec_restricted("from os import path", fs)
    assert result.startswith("RestrictedError"), result


def test_from_dotted_import_is_blocked(fs: SimulatedFilesystem) -> None:
    result = exec_restricted("from os.path import join", fs)
    assert result.startswith("RestrictedError"), result


def test_dunder_import_call_is_blocked(fs: SimulatedFilesystem) -> None:
    # __import__ is a valid Name; the Call-allowlist must reject it because
    # "__import__" isn't in ALLOWED_BUILTINS.
    result = exec_restricted("__import__('os')", fs)
    assert result.startswith("RestrictedError"), result


# ---------------------------------------------------------------------------
# Attribute-access attacks — ast.Attribute is not in ALLOWED_NODES.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "code",
    [
        "().__class__",
        "().__class__.__bases__",
        "[].__class__.__subclasses__()",
        "''.__class__",
        "{}.__class__",
    ],
)
def test_attribute_access_is_blocked(fs: SimulatedFilesystem, code: str) -> None:
    result = exec_restricted(code, fs)
    assert result.startswith("RestrictedError"), result


# ---------------------------------------------------------------------------
# Builtin-escape attacks — not in ALLOWED_BUILTINS.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "code",
    [
        "open('/etc/passwd')",
        "open('anything')",
        "exec('malicious')",
        "eval('1+1')",
        "compile('1', '<x>', 'exec')",
        "getattr([], '__class__')",
        "setattr([], 'y', 1)",
        "delattr([], 'y')",
        "globals()",
        "locals()",
        "vars()",
        "dir()",
        "hasattr([], 'y')",
        "input()",
        "print('x')",
    ],
)
def test_disallowed_builtin_calls_are_blocked(
    fs: SimulatedFilesystem, code: str
) -> None:
    result = exec_restricted(code, fs)
    assert result.startswith("RestrictedError"), result


# ---------------------------------------------------------------------------
# Subscript-escape attacks — ast.Subscript is not in ALLOWED_NODES.
# ---------------------------------------------------------------------------


def test_subscript_access_to_builtins_is_blocked(fs: SimulatedFilesystem) -> None:
    result = exec_restricted("__builtins__['open']", fs)
    assert result.startswith("RestrictedError"), result


def test_subscript_dunder_import_is_blocked(fs: SimulatedFilesystem) -> None:
    result = exec_restricted("__builtins__['__import__']", fs)
    assert result.startswith("RestrictedError"), result


def test_subscript_is_blocked_even_for_benign_indexing(
    fs: SimulatedFilesystem,
) -> None:
    # Section 7.1 omits ast.Subscript. That means even ``[1,2,3][0]`` is
    # blocked — the calculator workload doesn't need indexing. Documented
    # as intentional in the module docstring.
    result = exec_restricted("_result = [1, 2, 3][0]", fs)
    assert result.startswith("RestrictedError"), result


# ---------------------------------------------------------------------------
# Function/class definition attacks — blocked at the definition node.
# ---------------------------------------------------------------------------


def test_function_def_is_blocked(fs: SimulatedFilesystem) -> None:
    result = exec_restricted("def f(): pass", fs)
    assert result.startswith("RestrictedError"), result


def test_class_def_is_blocked(fs: SimulatedFilesystem) -> None:
    result = exec_restricted("class C: pass", fs)
    assert result.startswith("RestrictedError"), result


def test_bare_lambda_is_blocked(fs: SimulatedFilesystem) -> None:
    result = exec_restricted("lambda: 1", fs)
    assert result.startswith("RestrictedError"), result


def test_parameterized_lambda_is_blocked(fs: SimulatedFilesystem) -> None:
    result = exec_restricted("lambda x: x + 1", fs)
    assert result.startswith("RestrictedError"), result


def test_async_def_is_blocked(fs: SimulatedFilesystem) -> None:
    result = exec_restricted("async def f(): pass", fs)
    assert result.startswith("RestrictedError"), result


# ---------------------------------------------------------------------------
# Control-flow attacks — yield / yield from / await at module scope.
#
# In Python 3.11, ``yield 1``, ``yield from []``, and ``await x`` all parse
# successfully at module level; AST walk then yields ``Yield`` / ``YieldFrom``
# / ``Await`` nodes which are NOT in ALLOWED_NODES, so we expect
# ``RestrictedError`` (not ``SyntaxError``).
# ---------------------------------------------------------------------------


def test_yield_is_blocked(fs: SimulatedFilesystem) -> None:
    result = exec_restricted("yield 1", fs)
    assert result.startswith("RestrictedError"), result


def test_yield_from_is_blocked(fs: SimulatedFilesystem) -> None:
    result = exec_restricted("yield from []", fs)
    assert result.startswith("RestrictedError"), result


def test_await_is_blocked(fs: SimulatedFilesystem) -> None:
    result = exec_restricted("await x", fs)
    assert result.startswith("RestrictedError"), result


# ---------------------------------------------------------------------------
# Iteration cap — sys.settrace fires on runaway loops.
# ---------------------------------------------------------------------------


def test_iteration_cap_fires_on_while_true(fs: SimulatedFilesystem) -> None:
    result = exec_restricted("while True:\n    _result = 1", fs)
    assert result.startswith("RestrictedError"), result
    assert "Iteration cap exceeded" in result


def test_iteration_cap_fires_on_large_range(fs: SimulatedFilesystem) -> None:
    result = exec_restricted(
        "total = 0\nfor i in range(1000000):\n    total = total + 1\n_result = total",
        fs,
    )
    assert result.startswith("RestrictedError"), result
    assert "Iteration cap exceeded" in result


def test_recursion_is_structurally_impossible(fs: SimulatedFilesystem) -> None:
    # The restricted language can't express recursion because FunctionDef and
    # Lambda are both blocked. The test documents this: any attempt to define
    # a recursive function is rejected at the definition node.
    result = exec_restricted("def f():\n    f()\nf()", fs)
    assert result.startswith("RestrictedError"), result


# ---------------------------------------------------------------------------
# Valid calculator operations — must succeed.
# ---------------------------------------------------------------------------


def test_bare_expression_without_result_returns_string_none(
    fs: SimulatedFilesystem,
) -> None:
    # ``_result`` is seeded to None in the namespace, so a script that never
    # assigns to it surfaces as the string "None" (str(None)).
    assert exec_restricted("1 + 2", fs) == "None"


def test_simple_assignment_to_result(fs: SimulatedFilesystem) -> None:
    assert exec_restricted("_result = 1 + 2", fs) == "3"


def test_sum_builtin(fs: SimulatedFilesystem) -> None:
    assert exec_restricted("_result = sum([1, 2, 3])", fs) == "6"


def test_max_builtin(fs: SimulatedFilesystem) -> None:
    assert exec_restricted("_result = max(1, 5, 3)", fs) == "5"


def test_sorted_builtin(fs: SimulatedFilesystem) -> None:
    assert exec_restricted("_result = sorted([3, 1, 2])", fs) == "[1, 2, 3]"


def test_conditional_expression(fs: SimulatedFilesystem) -> None:
    # If-expressions parse as ast.IfExp — but Section 7.1 only lists ast.If
    # (statement form), not ast.IfExp. So this should actually be blocked.
    # The ticket's "Valid calculator use" list requests testing
    # ``_result = 5 if 1 > 0 else 10`` returns "5" — but IfExp is not in
    # ALLOWED_NODES, so the honest behavior is RestrictedError. We assert the
    # true behavior and flag this ambiguity in the report.
    result = exec_restricted("_result = 5 if 1 > 0 else 10", fs)
    assert result.startswith("RestrictedError"), result


def test_for_loop_accumulator(fs: SimulatedFilesystem) -> None:
    code = "total = 0\nfor i in range(5):\n    total = total + i\n_result = total"
    assert exec_restricted(code, fs) == "10"


def test_arithmetic_precedence(fs: SimulatedFilesystem) -> None:
    assert exec_restricted("_result = (5 + 3) * 2 ** 2", fs) == "32"


def test_while_loop_terminates_under_cap(fs: SimulatedFilesystem) -> None:
    code = "i = 0\nwhile i < 5:\n    i = i + 1\n_result = i"
    assert exec_restricted(code, fs) == "5"


def test_boolean_and_comparison(fs: SimulatedFilesystem) -> None:
    assert exec_restricted("_result = 1 < 2 and 3 > 2", fs) == "True"


def test_list_and_tuple_literals(fs: SimulatedFilesystem) -> None:
    assert exec_restricted("_result = sum([1, 2, 3]) + len([4, 5])", fs) == "8"


# ---------------------------------------------------------------------------
# Comprehensions — BLOCKED (Section 7.1 omission).
# ---------------------------------------------------------------------------


def test_list_comprehension_is_blocked(fs: SimulatedFilesystem) -> None:
    result = exec_restricted("_result = [x * 2 for x in range(10)]", fs)
    assert result.startswith("RestrictedError"), result


def test_set_comprehension_is_blocked(fs: SimulatedFilesystem) -> None:
    result = exec_restricted("_result = {x for x in range(5)}", fs)
    assert result.startswith("RestrictedError"), result


def test_dict_comprehension_is_blocked(fs: SimulatedFilesystem) -> None:
    result = exec_restricted("_result = {x: x for x in range(5)}", fs)
    assert result.startswith("RestrictedError"), result


def test_generator_expression_is_blocked(fs: SimulatedFilesystem) -> None:
    result = exec_restricted("_result = sum(x for x in range(5))", fs)
    assert result.startswith("RestrictedError"), result


# ---------------------------------------------------------------------------
# Edge cases on the input and error-string contract.
# ---------------------------------------------------------------------------


def test_empty_code_returns_string_none(fs: SimulatedFilesystem) -> None:
    assert exec_restricted("", fs) == "None"


def test_whitespace_only_returns_string_none(fs: SimulatedFilesystem) -> None:
    assert exec_restricted("   \n\t  ", fs) == "None"


def test_comment_only_returns_string_none(fs: SimulatedFilesystem) -> None:
    assert exec_restricted("# hello", fs) == "None"


def test_malformed_source_returns_syntax_error(fs: SimulatedFilesystem) -> None:
    result = exec_restricted("1 +", fs)
    assert result.startswith("SyntaxError"), result


def test_runtime_error_from_zero_division(fs: SimulatedFilesystem) -> None:
    result = exec_restricted("_result = 1 / 0", fs)
    assert result.startswith("RuntimeError"), result


# ---------------------------------------------------------------------------
# validate_ast direct usage — confirm it raises rather than returning.
# ---------------------------------------------------------------------------


def test_validate_ast_passes_for_clean_tree() -> None:
    tree = ast.parse("_result = 1 + 2", mode="exec")
    validate_ast(tree)  # must not raise


def test_validate_ast_raises_on_attribute_access() -> None:
    tree = ast.parse("x.y", mode="exec")
    with pytest.raises(RestrictedPythonError):
        validate_ast(tree)


def test_validate_ast_raises_on_disallowed_call() -> None:
    tree = ast.parse("open('x')", mode="exec")
    with pytest.raises(RestrictedPythonError):
        validate_ast(tree)


def test_validate_ast_indirect_call_error_message_mentions_call_shape() -> None:
    # A call whose func is not an ast.Name (e.g. attribute call) must be
    # rejected with the "Only direct builtin calls allowed" message. Because
    # the ast.walk order visits parents before children and Attribute is
    # ALSO disallowed, the first raise is actually the Attribute node. Assert
    # we get *some* RestrictedPythonError either way.
    tree = ast.parse("[].append(1)", mode="exec")
    with pytest.raises(RestrictedPythonError):
        validate_ast(tree)


# ---------------------------------------------------------------------------
# Tracer cleanup — sys.settrace must be None after every exec, success or
# failure. Leaking a tracer would wreck the test runner.
# ---------------------------------------------------------------------------


def test_tracer_uninstalled_after_success(fs: SimulatedFilesystem) -> None:
    import sys as _sys

    exec_restricted("_result = 1 + 1", fs)
    assert _sys.gettrace() is None


def test_tracer_uninstalled_after_iteration_cap(fs: SimulatedFilesystem) -> None:
    import sys as _sys

    exec_restricted("while True:\n    _result = 1", fs)
    assert _sys.gettrace() is None


def test_tracer_uninstalled_after_runtime_error(fs: SimulatedFilesystem) -> None:
    import sys as _sys

    exec_restricted("_result = 1 / 0", fs)
    assert _sys.gettrace() is None


# ---------------------------------------------------------------------------
# TOOLS wiring — run_python is callable and routes through exec_restricted.
# ---------------------------------------------------------------------------


def test_tools_run_python_is_callable() -> None:
    assert callable(TOOLS["run_python"])


def test_tools_run_python_computes_result(fs: SimulatedFilesystem) -> None:
    assert TOOLS["run_python"](fs, "_result = 1 + 1") == "2"


def test_tools_run_python_blocks_import(fs: SimulatedFilesystem) -> None:
    result = TOOLS["run_python"](fs, "import os")
    assert "RestrictedError" in result


def test_tools_run_python_blocks_open(fs: SimulatedFilesystem) -> None:
    result = TOOLS["run_python"](fs, "open('/etc/passwd')")
    assert "RestrictedError" in result


def test_tools_submit_answer_is_still_none_placeholder() -> None:
    assert TOOLS["submit_answer"] is None
