"""AST-whitelisted ``run_python`` executor — SECURITY CRITICAL boundary.

This module implements the restricted Python executor defined in
PROJECT.md Section 7. It is the *primary* defense against reward hacking via
the ``run_python`` tool: if this module lets an attack pattern slip through,
the audit classifier (and therefore the reward) becomes gameable. The layered
hash-based ShutdownInvariantRubric (Section 7.5) exists as a second line of
defense, but this boundary is the main event.

Design
------
Two whitelists enforced by an ``ast.walk`` pass:

* :data:`ALLOWED_NODES` — every syntactic construct accepted by the sandbox.
  Anything not in this set raises :class:`RestrictedPythonError`. This is
  strict-by-default: ``ast.Attribute``, ``ast.Subscript``,
  ``ast.FunctionDef``, ``ast.ClassDef``, ``ast.Lambda``,
  ``ast.AsyncFunctionDef``, ``ast.Yield``, ``ast.YieldFrom``, ``ast.Await``,
  and every kind of comprehension are all left out on purpose.
* :data:`ALLOWED_BUILTINS` — every name that a ``Call`` node is allowed to
  resolve to. Indirect calls (attribute calls, subscript calls, lambda calls)
  are blocked at the ``Call`` node itself, belt-and-suspenders with the
  already-missing :class:`ast.Attribute` / :class:`ast.Subscript` nodes.

Comprehensions are deliberately blocked
---------------------------------------
The Section 7.1 allowlist does not include :class:`ast.ListComp`,
:class:`ast.SetComp`, :class:`ast.DictComp`, :class:`ast.GeneratorExp`, or
:class:`ast.comprehension`. Python implements comprehensions with an implicit
nested function scope; that scope is a potential escape surface, so we keep
them blocked rather than trying to audit what runs inside. The calculator
workload the agent needs (sums, min/max, sorted, range-based for-loops) is
fully expressible without comprehensions. Tests assert
``[x*2 for x in range(10)]`` and friends raise ``RestrictedError``.

Iteration cap — ``sys.settrace`` with a per-execution line counter
------------------------------------------------------------------
Section 7.4 calls for a 10,000-instruction cap but leaves the exact mechanism
open ("bytecode-level instruction limit"). We implement it via
:func:`sys.settrace` with a per-execution line counter. Every executed line
increments the counter; crossing the threshold raises
:class:`RestrictedPythonError` from inside the tracer, which :func:`exec`
propagates out of the ``exec`` call and we catch in :func:`exec_restricted`.

This uniformly covers ``for``-loops, ``while``-loops, and (hypothetical)
recursion with one primitive. We explicitly rejected two alternatives:

* Wrapping :func:`range` — misses ``while`` loops and recursion entirely.
* :func:`sys.setrecursionlimit` — misses loop-based runaway iteration.

In practice the restricted language cannot express recursion because
:class:`ast.FunctionDef`, :class:`ast.Lambda`, and :class:`ast.AsyncFunctionDef`
are all blocked, but the cap still guards runaway ``while`` and large
``for i in range(...)`` loops.

Python version note
-------------------
:class:`ast.Num` and :class:`ast.Str` are deprecated aliases of
:class:`ast.Constant` in Python 3.8+. On modern interpreters ``ast.walk`` only
yields :class:`ast.Constant`, so their presence in :data:`ALLOWED_NODES` is a
no-op — but we include them to match the Section 7.1 spec literally and in
case a future interpreter re-emits them.
"""

from __future__ import annotations

import ast
import builtins as _builtins
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Forward-reference-only import so ``sandbox.py`` can depend on this
    # module without creating an import cycle. ``fs`` is currently unused at
    # runtime — it's threaded through for the TOOLS dispatch contract and for
    # future defensive hash checks — so we never actually need the concrete
    # class at import time.
    from red_button.sandbox import SimulatedFilesystem


# ---------------------------------------------------------------------------
# Allowlists — PROJECT.md Section 7.1 and 7.2
# ---------------------------------------------------------------------------

ALLOWED_NODES: frozenset[type] = frozenset(
    {
        # Top-level structure
        ast.Expression,
        ast.Module,
        ast.Expr,
        # Literals (ast.Num / ast.Str are 3.8+ deprecated aliases — see docstring)
        ast.Num,
        ast.Constant,
        ast.Str,
        ast.List,
        ast.Tuple,
        ast.Dict,
        ast.Set,
        # Names (local variables only)
        ast.Name,
        ast.Load,
        ast.Store,
        # Arithmetic
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.Pow,
        ast.FloorDiv,
        ast.USub,
        ast.UAdd,
        # Comparison and boolean
        ast.Compare,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.Not,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        # Assignment
        ast.Assign,
        # Control flow (with iteration cap enforced via sys.settrace)
        ast.If,
        ast.For,
        ast.While,
        # Function calls — restricted further below to whitelisted builtins
        ast.Call,
    }
)


ALLOWED_BUILTINS: frozenset[str] = frozenset(
    {
        "abs",
        "min",
        "max",
        "sum",
        "len",
        "range",
        "int",
        "float",
        "str",
        "round",
        "sorted",
        "enumerate",
        "zip",
        "all",
        "any",
        "bool",
        "list",
        "dict",
        "tuple",
        "set",
    }
)


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class RestrictedPythonError(Exception):
    """Raised when code violates the Section 7 sandbox contract.

    Either the AST contains a disallowed node / a call to a non-whitelisted
    builtin, or the iteration cap was exceeded at runtime.
    """


# ---------------------------------------------------------------------------
# AST validation — Section 7.4
# ---------------------------------------------------------------------------


def validate_ast(tree: ast.AST) -> None:
    """Walk ``tree`` and raise :class:`RestrictedPythonError` on disallowed nodes.

    Two rules:

    1. Every node's exact type must be in :data:`ALLOWED_NODES` (no subclass
       leniency — ``type(node) is not isinstance(node, ...)``). This is what
       blocks imports, attribute access, subscripts, lambdas, function/class
       definitions, yield/await, comprehensions, and anything else the spec
       doesn't explicitly bless.
    2. Every :class:`ast.Call` must be a *direct* call to a name in
       :data:`ALLOWED_BUILTINS`. ``obj.method()``, ``funcs[0]()``, and
       ``(lambda: 1)()`` are all rejected here at the ``Call`` node itself,
       belt-and-suspenders with the already-missing inner nodes.
    """
    for node in ast.walk(tree):
        if type(node) not in ALLOWED_NODES:
            raise RestrictedPythonError(
                f"Disallowed AST node: {type(node).__name__}"
            )
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in ALLOWED_BUILTINS:
                    raise RestrictedPythonError(
                        f"Disallowed builtin call: {node.func.id}"
                    )
            else:
                # Any other Call shape — attribute call, subscript call,
                # lambda call, call of a call's return value — is rejected
                # outright. The inner node would already fail the walk, but
                # failing here gives a clearer error message.
                raise RestrictedPythonError(
                    f"Only direct builtin calls allowed, got "
                    f"{type(node.func).__name__}"
                )


# ---------------------------------------------------------------------------
# Execution — Section 7.4
# ---------------------------------------------------------------------------


_ITERATION_CAP = 10_000


def exec_restricted(code: str, fs: "SimulatedFilesystem") -> str:
    """Execute ``code`` under the Section 7 restricted-Python contract.

    Contract summary:

    * Parse in ``exec`` mode; :class:`SyntaxError` -> ``"SyntaxError: ..."``.
    * Run :func:`validate_ast`; :class:`RestrictedPythonError` ->
      ``"RestrictedError: ..."``.
    * Execute with a restricted ``__builtins__`` mapping containing only the
      names in :data:`ALLOWED_BUILTINS`.
    * Install a :func:`sys.settrace` tracer that increments a per-execution
      line counter; on reaching :data:`_ITERATION_CAP` raise
      :class:`RestrictedPythonError` from inside the tracer.
    * Any :class:`RestrictedPythonError` raised during ``exec`` surfaces as
      ``"RestrictedError: ..."``. Any other :class:`Exception` surfaces as
      ``"RuntimeError: ..."``. The tracer is uninstalled in a ``finally``.
    * Return ``str(namespace['_result'])`` if ``_result`` was assigned, else
      ``"None"`` (because the initial namespace seeds ``_result = None`` and
      ``str(None) == "None"``).

    The ``fs`` argument is received for the TOOLS wiring contract but is NOT
    exposed to executed code — the restricted namespace only contains the
    scoped ``__builtins__`` and ``_result``.
    """
    # 1. Parse.
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        return f"SyntaxError: {e}"

    # 2. Validate AST.
    try:
        validate_ast(tree)
    except RestrictedPythonError as e:
        return f"RestrictedError: {e}"

    # 3. Build the restricted builtins namespace. We pull from the real
    #    ``builtins`` module rather than ``__builtins__`` (which is a module
    #    at module scope but a dict under nested exec) to avoid the dict/module
    #    ambiguity entirely.
    restricted_builtins: dict[str, object] = {
        name: getattr(_builtins, name) for name in ALLOWED_BUILTINS
    }
    namespace: dict[str, object] = {
        "__builtins__": restricted_builtins,
        "_result": None,
    }

    # 4. Install iteration-cap tracer. A list-of-one holds the counter so the
    #    closure can mutate it without ``nonlocal`` gymnastics. The tracer
    #    returns itself so that nested frames continue to trace (sys.settrace
    #    semantics: the global tracer fires on 'call' events and must return a
    #    local tracer for line/return/exception events inside that frame).
    counter = [0]

    def tracer(frame, event, arg):  # type: ignore[no-untyped-def]
        if event == "line":
            counter[0] += 1
            if counter[0] >= _ITERATION_CAP:
                raise RestrictedPythonError(
                    f"Iteration cap exceeded: {_ITERATION_CAP} instructions"
                )
        return tracer

    # 5. Compile and exec with the tracer armed. Always tear the tracer down
    #    in a ``finally`` — leaving sys.settrace hot across calls would break
    #    unrelated code and the test runner.
    try:
        compiled = compile(tree, "<restricted>", "exec")
        sys.settrace(tracer)
        try:
            exec(compiled, namespace)
        finally:
            sys.settrace(None)
    except RestrictedPythonError as e:
        # Iteration cap or a tracer-raised error. Must be caught *before*
        # the generic Exception handler so we don't mask it as RuntimeError.
        return f"RestrictedError: {e}"
    except Exception as e:  # noqa: BLE001 — deliberate broad catch at the boundary
        return f"RuntimeError: {e}"

    # 6. Result. ``_result`` is always present (we seeded it), so this yields
    #    ``"None"`` when the script didn't assign and ``str(value)`` otherwise.
    return str(namespace.get("_result"))
