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


# ------------------------------------------------- 模型卡里的技术报告
# 真实形态：Qwen3.8-Flash-Next 的技术报告只有 GitHub 上的 PDF，没上 arXiv；
# GLM-5.3-Flash 反过来，只在 HF tag 里挂了一篇 2026-02 的旧论文。
QWEN_CARD = """## Qwen3.8-Flash-Next

Join our [Discord](https://discord.gg/qwen) or scan the [WeChat QR](https://qwen.ai/wechat).
For more details, please refer to [the technical report](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf).
Try it online at [chat.qwen.ai](https://chat.qwen.ai).

```python
from transformers import AutoModelForCausalLM
```
"""


def test_technical_report_pdf_in_model_card_is_found():
    """Qwen3.8-Flash-Next 的技术报告只挂 GitHub，没上 arXiv。

    只查 arXiv 的话这类发布会整个漏掉——这正是加这段扫描的原因。
    """
    found = radar.extract_report_links(QWEN_CARD, [])

    assert found["docs"] == [
        "https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf"]
    assert found["arxiv"] == []


def test_arxiv_id_is_read_from_hf_tags():
    """HF 把关联论文放进结构化 tag，比拿正则啃 README 可靠。"""

    found = radar.extract_report_links("", ["safetensors", "arxiv:2602.15763", "license:mit"])
    assert found["arxiv"] == ["2602.15763"]


def test_community_and_demo_links_are_not_mistaken_for_reports():
    """模型卡里满是 Discord／微信／在线体验链接，收进来会淹掉真信号。"""

    docs = " ".join(radar.extract_report_links(QWEN_CARD, [])["docs"])
    assert "discord" not in docs
    assert "chat.qwen.ai" not in docs
    assert "wechat" not in docs


def test_arxiv_month_is_surfaced_so_stale_papers_are_obvious():
    """GLM-5.3-Flash 挂的是 2026-02 的旧论文，不是本周新报告。

    只显示一个 ID 的话，读者会默认它是随这次发布来的。
    """
    assert radar.arxiv_month("2602.15763") == "2026-02"
    assert radar.arxiv_month("2608.24949") == "2026-08"
    assert radar.arxiv_month("garbage") == ""


def _vendor(reports):
    return {"name": "阿里 Qwen", "org": "Qwen", "kb_entry": "README.md",
            "models": [{"id": "Qwen/Qwen3.8-Flash-Next", "created": "2026-08-24",
                        "downloads": 0, "reports": reports}]}


def test_report_surfaces_model_card_technical_reports():
    text = radar.render([], [_vendor(
        {"arxiv": ["2602.15763"], "docs": ["https://x.test/tech_report.pdf"]})],
        [], [], 7, "2026-08-28")

    assert "tech_report.pdf" in text
    assert "2602.15763" in text
    assert "2026-02" in text          # 年月标出来，旧论文一眼可见


def test_model_with_no_report_says_so_rather_than_staying_silent():
    """『查了没找到』和『压根没查』必须能区分——后者是盲区，前者是事实。"""

    text = radar.render([], [_vendor({"arxiv": [], "docs": []})], [], [], 7, "2026-08-28")
    assert "未找到技术报告" in text


def test_bare_paper_link_is_a_report_but_wallpaper_is_not():
    """模型卡常写成 `[paper](...)`，这类必须收；而词边界要挡住 wallpaper
    这种子串误命中。"""

    found = radar.extract_report_links(
        "See the [paper](https://x.test/report.pdf) and the "
        "[banner](https://x.test/wallpaper.png).", [])
    assert found["docs"] == ["https://x.test/report.pdf"]
