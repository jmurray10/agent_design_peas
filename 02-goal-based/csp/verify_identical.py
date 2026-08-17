"""Check mechanically that after.py reuses before.py's solver instead of copying it.

The README claims the solver is byte-for-byte identical across before.py and
after.py. A claim like that is worth exactly as much as its check, so this script
makes it a check.

Three levels, cheapest first:

1. Static. Parse after.py with `ast` and confirm it binds no name `CSP` and no name
   `backtracking_search` of its own -- no class, no function, no assignment. Parsing
   is used rather than grep because grep cannot tell a definition from a mention in
   a comment, and a claim this load-bearing should not rest on a substring.
2. Import graph. Confirm after.py takes both names from `before`.
3. Runtime. Import both modules and confirm the objects are identical, and that the
   source Python attributes them to before.py.

Prints a SHA-256 of the solver source as it exists in before.py. There is one copy
of that text in this directory, and this is its digest.

Exits non-zero if any check fails.
"""

import ast
import hashlib
import inspect
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SOLVER_NAMES = ("CSP", "backtracking_search")

_failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
    if not ok:
        _failures.append(label)


def bound_names(path: Path) -> dict[str, str]:
    """Every name the module binds itself, mapped to how it binds it.

    Imports are excluded on purpose: importing a name is the behaviour under test,
    while defining or assigning one is the behaviour being ruled out.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    bindings: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            bindings[node.name] = "class definition"
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bindings[node.name] = "function definition"
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bindings[target.id] = "assignment"
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            bindings[node.target.id] = "annotated assignment"
    return bindings


def names_imported_from(path: Path, module: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == module:
            found.update(alias.asname or alias.name for alias in node.names)
    return found


def main() -> int:
    after_py, before_py = HERE / "after.py", HERE / "before.py"

    print("Static check: after.py defines no solver of its own")
    after_bindings = bound_names(after_py)
    for name in SOLVER_NAMES:
        how = after_bindings.get(name)
        check(f"after.py does not bind {name!r} itself",
              how is None,
              "" if how is None else f"found: {how}")

    print("\nStatic check: before.py is where the solver is defined")
    before_bindings = bound_names(before_py)
    for name in SOLVER_NAMES:
        check(f"before.py defines {name!r}",
              name in before_bindings,
              before_bindings.get(name, "not found"))

    print("\nImport check: after.py takes the solver from before.py")
    imported = names_imported_from(after_py, "before")
    for name in SOLVER_NAMES:
        check(f"after.py imports {name!r} from before", name in imported)

    print("\nRuntime check: the objects are the same objects")
    try:
        import after
        import before
    except Exception as exc:  # a module that will not import cannot be verified
        check("after.py and before.py import cleanly", False, f"{type(exc).__name__}: {exc}")
        print(f"\nFAIL: {len(_failures)} check(s) failed: {', '.join(_failures)}")
        return 1

    for name in SOLVER_NAMES:
        after_obj, before_obj = getattr(after, name), getattr(before, name)
        check(f"after.{name} is before.{name}", after_obj is before_obj)
        source_file = Path(inspect.getsourcefile(after_obj) or "")
        check(f"Python attributes after.{name} to before.py",
              source_file.name == "before.py",
              source_file.name)

    solver_source = inspect.getsource(before.CSP) + inspect.getsource(before.backtracking_search)
    digest = hashlib.sha256(solver_source.encode("utf-8")).hexdigest()

    print()
    if _failures:
        print(f"FAIL: {len(_failures)} check(s) failed: {', '.join(_failures)}")
        return 1
    print("PASS: after.py does not contain a solver. It imports the one in before.py.")
    print(f"      solver source: {len(solver_source.splitlines())} lines in {before_py}")
    print(f"      sha256: {digest}")
    print("      One copy of that text exists in this directory. This is its digest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
