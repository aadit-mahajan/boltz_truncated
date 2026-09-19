"""The Colab notebook: valid, clean, and only using options that exist."""

import ast
import json
import re
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/boltz2_optimized_colab.ipynb"


@pytest.fixture(scope="module")
def notebook():
    return json.loads(NOTEBOOK.read_text())


@pytest.fixture(scope="module")
def code_cells(notebook):
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def test_notebook_is_a_clean_nbformat_4_document(notebook):
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        assert cell["cell_type"] in {"code", "markdown"}
        assert isinstance(cell["source"], list)
        if cell["cell_type"] == "code":
            # Committed outputs make diffs unreadable and can leak paths.
            assert cell["outputs"] == []
            assert cell["execution_count"] is None


def test_every_code_cell_is_valid_python(code_cells):
    for index, cell in enumerate(code_cells):
        source = "".join(cell["source"])
        # Colab form cells carry #@param annotations, which are plain comments.
        ast.parse(source)  # raises SyntaxError naming the cell's content


def test_cells_do_not_rely_on_shell_or_magic_lines(code_cells):
    """Everything runs through subprocess, so the notebook also runs under nbconvert."""
    for cell in code_cells:
        for line in cell["source"]:
            assert not line.lstrip().startswith(("!", "%")), line


def test_referenced_cli_options_exist(code_cells):
    from click.testing import CliRunner

    from boltz.main import cli

    documented = set(re.findall(r"--[a-z][a-z_]+", CliRunner().invoke(cli, ["predict", "--help"]).output))
    used = set()
    for cell in code_cells:
        source = "".join(cell["source"])
        if '"-m", "boltz.main", "predict"' in source or "boltz.main" in source:
            used |= set(re.findall(r'"(--[a-z][a-z_]+)"', source))
    # Options passed to the comparison and setup scripts are checked separately.
    used -= {"--reference-seed", "--candidate-seed", "--structure-rmsd-atol",
             "--atol", "--rtol", "--exact"}
    assert used, "no boltz predict options found in the notebook"
    assert used <= documented, f"notebook uses options predict does not define: {sorted(used - documented)}"


def test_referenced_repository_paths_exist(code_cells):
    root = NOTEBOOK.resolve().parents[1]
    referenced = set()
    for cell in code_cells:
        referenced |= set(re.findall(r'RUN_REPO / "([^"]+)"', "".join(cell["source"])))
    assert referenced
    for relative in referenced:
        assert (root / relative).exists(), f"notebook points at a missing path: {relative}"


def test_profiles_named_in_the_notebook_are_real(code_cells):
    from boltz import opt

    named = set()
    for cell in code_cells:
        named |= set(re.findall(r'profile="([a-z]+)"', "".join(cell["source"])))
        named |= set(re.findall(r'for profile in \(([^)]*)\)', "".join(cell["source"])))
    named = {part.strip().strip('"') for chunk in named for part in chunk.split(",")} - {""}
    assert named, "no profile names found"
    assert named <= set(opt.PROFILES), sorted(named - set(opt.PROFILES))


def test_levers_ablated_in_the_notebook_are_real(code_cells):
    from boltz import opt

    for cell in code_cells:
        source = "".join(cell["source"])
        match = re.search(r'for lever in \(([^)]*)\)', source)
        if match:
            levers = {part.strip().strip('"') for part in match[1].split(",") if part.strip()}
            assert levers <= opt.PROFILES["fast"], sorted(levers - opt.PROFILES["fast"])
            return
    pytest.fail("the notebook no longer ablates individual levers")
