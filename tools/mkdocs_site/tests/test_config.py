import shutil
from pathlib import Path

import pytest
import yaml

from tools.mkdocs_site.config import ConfigError, write_generated_config
from tools.mkdocs_site.inventory import scan_inventory
from tools.mkdocs_site.models import BuildPaths
from tools.mkdocs_site.staging import stage_wiki


BASE_CONFIG = """\
site_name: Test Knowledge Wiki
site_url: https://suhaibo666.github.io/llm-knowledge/
docs_dir: .mkdocs-cache/docs
site_dir: site
use_directory_urls: false
theme:
  name: material
plugins:
  - search
"""


def make_repo(tmp_path: Path, fixture_wiki: Path) -> BuildPaths:
    repo = tmp_path / "repo"
    shutil.copytree(fixture_wiki, repo / "wiki")
    (repo / "mkdocs.yml").write_text(BASE_CONFIG, encoding="utf-8")
    return BuildPaths.from_repo(repo)


def test_generated_config_inherits_base_and_targets_absolute_build_paths(
    tmp_path: Path, fixture_wiki: Path
) -> None:
    paths = make_repo(tmp_path, fixture_wiki)
    inventory = scan_inventory(paths.wiki)
    stage_result = stage_wiki(paths, inventory)

    generated_path = write_generated_config(paths, inventory, stage_result)

    generated = yaml.safe_load(generated_path.read_text(encoding="utf-8"))
    assert generated_path == paths.generated_config
    assert generated["site_name"] == "Test Knowledge Wiki"
    assert generated["site_url"] == "https://suhaibo666.github.io/llm-knowledge/"
    assert generated["use_directory_urls"] is False
    assert generated["theme"] == {"name": "material"}
    assert generated["plugins"] == ["search"]
    assert generated["docs_dir"] == str(paths.staging.resolve())
    assert generated["site_dir"] == str(paths.site.resolve())
    assert generated["nav"] == [
        {"LLM Knowledge Wiki": "index.md"},
        {
            "工程实现 — 知识地图": [
                {"工程实现 — 知识地图": "domain/index.md"},
                {"10_article": "domain/10_article.md"},
            ]
        },
    ]


def test_generated_config_rejects_changed_base_path_contract(
    tmp_path: Path, fixture_wiki: Path
) -> None:
    paths = make_repo(tmp_path, fixture_wiki)
    (paths.repo / "mkdocs.yml").write_text(
        BASE_CONFIG.replace("docs_dir: .mkdocs-cache/docs", "docs_dir: wiki"),
        encoding="utf-8",
    )
    inventory = scan_inventory(paths.wiki)
    stage_result = stage_wiki(paths, inventory)

    with pytest.raises(ConfigError, match=r"docs_dir.*\.mkdocs-cache/docs"):
        write_generated_config(paths, inventory, stage_result)

    assert not paths.generated_config.exists()
