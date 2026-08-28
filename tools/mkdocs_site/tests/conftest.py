from pathlib import Path
from shutil import copytree

import pytest


@pytest.fixture
def fixture_wiki(tmp_path: Path) -> Path:
    source = Path(__file__).parent / "fixtures" / "wiki"
    destination = tmp_path / "wiki"
    copytree(source, destination)
    return destination
