"""上游雷达的行为测试。全部离线：只测纯函数与渲染，不打网络。"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import radar


REPO_ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ watchlist
def test_watchlist_is_valid_and_every_entry_is_actionable():
    """清单是雷达的唯一输入，缺字段会静默产出空结果，所以逐项校验。"""

    data = radar.load_watchlist()
    assert data["repos"] and data["vendors"] and data["arxiv"]

    for entry in data["repos"]:
        assert entry["repo"].count("/") == 1, entry
        assert entry.get("branch"), entry
        assert entry.get("host", "github") in radar.HOST_URL, entry
        # kb_entry 要指向真实存在的页面，否则报告里的链接是死的
        assert (REPO_ROOT / entry["kb_entry"]).exists(), entry["kb_entry"]

    for entry in data["vendors"]:
        assert entry.get("org"), entry
        assert (REPO_ROOT / entry["kb_entry"]).exists(), entry["kb_entry"]

    for entry in data["arxiv"]:
        # 每条 query 必须带 LLM 约束：不带的话 MoE / attention 这类词
        # 会把视频理解、深度伪造、加密货币预测全捞进来
        assert "language model" in entry["query"] or "LLM" in entry["query"], entry


# ---------------------------------------------------------------------- state
def test_state_round_trips(tmp_path):
    path = tmp_path / "state.json"
    assert radar.load_state(path) == {"repos": {}, "vendors": {}, "arxiv": {}}

    state = {"repos": {"github:a/b": {"head": "abc"}}, "vendors": {}, "arxiv": {}}
    radar.save_state(state, path)
    assert radar.load_state(path) == state


def test_parse_iso_handles_both_forms_and_bad_input():
    assert radar.parse_iso("2026-08-26T10:00:00Z").year == 2026
    assert radar.parse_iso("2026-08-26T10:00:00+00:00").year == 2026
    # 无时区的输入要被当作 UTC，否则和 since 比较时会抛 naive/aware 异常
    assert radar.parse_iso("2026-08-26T10:00:00").tzinfo is not None
    assert radar.parse_iso("not-a-date").year == datetime.min.year


# --------------------------------------------------------------------- repos
def _entry(**over):
    base = {"name": "demo", "repo": "org/demo", "branch": "main",
            "kb_baseline": None, "kb_entry": "README.md", "category": "测试"}
    base.update(over)
    return base


def test_first_run_does_not_report_every_tag_as_new(monkeypatch):
    """首跑时仓库所有 tag 都是『没见过』的，那不是信号。"""

    monkeypatch.setattr(radar, "ls_remote", lambda url, pattern: (
        {"refs/heads/main": "a" * 40} if "heads" in pattern
        else {"refs/tags/v1.0": "b" * 40, "refs/tags/v2.0": "c" * 40}
    ))
    state = {"repos": {}, "vendors": {}, "arxiv": {}}
    finding = radar.check_repo(_entry(), state, [])

    assert finding["new_tags"] == []
    assert finding["moved"] is False          # 没有上一次记录，不算「变化」
    assert state["repos"]["github:org/demo"]["tags"] == ["v1.0", "v2.0"]

    # 第二次：只有真正新增的 tag 才报
    monkeypatch.setattr(radar, "ls_remote", lambda url, pattern: (
        {"refs/heads/main": "d" * 40} if "heads" in pattern
        else {"refs/tags/v1.0": "b" * 40, "refs/tags/v2.0": "c" * 40,
              "refs/tags/v3.0": "e" * 40}
    ))
    second = radar.check_repo(_entry(), state, [])
    assert second["new_tags"] == ["v3.0"]
    assert second["moved"] is True


def test_repo_failure_is_collected_not_fatal(monkeypatch):
    """一个源挂掉不能让整轮采集失败——否则周报会整期消失。"""

    def boom(url, pattern):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(radar, "ls_remote", boom)
    errors = []
    assert radar.check_repo(_entry(), {"repos": {}}, errors) is None
    assert errors and "connection reset" in errors[0]


def test_drift_is_only_computed_when_a_baseline_is_pinned(monkeypatch):
    monkeypatch.setattr(radar, "ls_remote", lambda url, pattern: (
        {"refs/heads/main": "f" * 40} if "heads" in pattern else {}
    ))
    called = []
    monkeypatch.setattr(radar, "compare_commits",
                        lambda *a: called.append(a) or {"ahead_by": 3, "titles": []})

    radar.check_repo(_entry(kb_baseline=None), {"repos": {}}, [])
    assert called == []                                    # 没钉基线就不查

    radar.check_repo(_entry(kb_baseline="0123abc"), {"repos": {}}, [])
    assert called and called[0][1] == "0123abc"

    # 基线就是当前 HEAD 时也不该去查
    called.clear()
    radar.check_repo(_entry(kb_baseline="f" * 40), {"repos": {}}, [])
    assert called == []


# -------------------------------------------------------------------- report
def _finding(**over):
    base = {"name": "vLLM", "category": "推理框架", "repo": "vllm-project/vllm",
            "host": "github", "branch": "main", "head": "a" * 40, "moved": True,
            "new_tags": [], "kb_baseline": "b" * 40,
            "kb_entry": "wiki/index.md", "drift": None}
    base.update(over)
    return base


def test_report_leads_with_stale_baselines():
    """『我们的页面落后了多少』是这份报告最该先说的事。"""

    stale = _finding(drift={"ahead_by": 229, "titles": ["fix: something"]})
    text = radar.render([stale], [], [], [], 7, "2026-08-26")

    assert text.index("KB 基线已过期") < text.index("仓库活动")
    assert "229 个提交" in text
    assert "fix: something" in text


def test_report_states_plainly_when_nothing_moved():
    text = radar.render([_finding(drift=None)], [], [], [], 7, "2026-08-26")
    assert "本期没有仓库的 KB 基线落后" in text
    assert "本期无新模型" in text
    assert "本期无新论文命中" in text


def test_collection_failures_are_surfaced_not_hidden():
    """静默吞掉失败会让读者以为『本期无变化』。"""

    text = radar.render([], [], [], ["仓库 X：timeout"], 7, "2026-08-26")
    assert "不代表它们没有变化" in text
    assert "仓库 X：timeout" in text


def test_truncated_paper_lists_say_so():
    group = {"name": "MoE", "papers": [
        {"id": "2608.1", "date": "2026-08-25", "title": "T"}], "truncated": 12}
    text = radar.render([], [], [group], [], 7, "2026-08-26")
    assert "另有 12 篇命中未列出" in text


def test_report_never_claims_to_have_written_wiki_pages():
    """雷达的边界就是不写分析页，报告头部必须把这点讲明。"""

    text = radar.render([], [], [], [], 7, "2026-08-26")
    assert "只报告事实" in text
    assert "source-faithful-analysis" in text


def test_release_tags_drops_ci_noise_and_sorts_naturally():
    """pytorch 有 6600+ 个 tag，其中 2751 个 viable/strict、1000+ 个 ciflow。

    全存进 state.json 会让它到 286KB 且每周提交，而这些对
    「上游发了什么版本」毫无信息量。
    """

    refs = {
        "refs/tags/v2.9.0": "a",
        "refs/tags/v2.10.0": "b",
        "refs/tags/v2.9.0^{}": "c",          # peeled ref，去重后应只算一个
        "refs/tags/v2.9.1-rc1": "d",
        "refs/tags/v2.1.0_core_r0.12.1": "e",  # MindSpeed 的真实形态
        "refs/tags/viable/strict/12345": "f",
        "refs/tags/ciflow/trunk/999": "g",
        "refs/tags/ykarnati-submodule-cc897f011": "h",
    }
    tags = radar.release_tags(refs)

    assert "viable/strict/12345" not in tags
    assert "ciflow/trunk/999" not in tags
    assert "ykarnati-submodule-cc897f011" not in tags
    assert "v2.1.0_core_r0.12.1" in tags
    # 自然序：2.9 必须排在 2.10 之前（字典序会反过来）
    assert tags.index("v2.9.0") < tags.index("v2.10.0")


def test_tracked_tags_are_capped():
    refs = {"refs/tags/v1.%d.0" % i: str(i) for i in range(300)}
    assert len(radar.release_tags(refs)) == radar.MAX_TRACKED_TAGS
