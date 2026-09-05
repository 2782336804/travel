from pathlib import Path
import sys


# 允许测试文件直接导入 backend/app 下的模块。
CURRENT_FILE = Path(__file__).resolve()
BACKEND_DIR = CURRENT_FILE.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import app.rag.retriever as retriever  # noqa: E402


EMPTY_USAGE = {"prompt_tokens": 0, "completion_tokens": 0}


async def test_retrieve_travel_guide_formats_chunks_as_text(monkeypatch) -> None:
    """测试 retriever 会把检索结果格式化成可直接引用的文本片段。"""

    async def fake_search_guide_chunks_with_usage(
        query: str, top_k: int = 3
    ) -> tuple[list[dict[str, str]], dict[str, int]]:
        assert query == "大理 古城 美食"
        assert top_k == 8
        return [
            {
                "source": "dali_guide.md",
                "title": "大理古城",
                "text": "大理古城适合慢游和拍照。",
            }
        ], EMPTY_USAGE

    async def fake_rerank_guide_chunks(query, matched_chunks, top_k, destination=None):
        return matched_chunks[:top_k], EMPTY_USAGE

    monkeypatch.setattr(retriever, "search_guide_chunks_with_usage", fake_search_guide_chunks_with_usage)
    monkeypatch.setattr(retriever, "rerank_guide_chunks", fake_rerank_guide_chunks)
    monkeypatch.setattr(retriever, "get_cached_json", async_fake_get_cached)
    monkeypatch.setattr(retriever, "set_cached_json", async_fake_set_cached)

    results, _, _ = await retriever.retrieve_travel_guide("大理 古城 美食", top_k=2)

    assert results == ["[来源: dali_guide.md | 标题: 大理古城]\n大理古城适合慢游和拍照。"]


async def test_retrieve_travel_guide_returns_empty_when_no_chunks(monkeypatch) -> None:
    """测试没有召回任何片段时，会返回空列表。"""

    async def fake_search_guide_chunks_with_usage(
        query: str, top_k: int = 3
    ) -> tuple[list[dict[str, str]], dict[str, int]]:
        assert query == "火星 沙漠 极地科考"
        assert top_k == 6
        return [], EMPTY_USAGE

    async def fake_rerank_guide_chunks(query, matched_chunks, top_k, destination=None):
        return matched_chunks[:top_k], EMPTY_USAGE

    monkeypatch.setattr(retriever, "search_guide_chunks_with_usage", fake_search_guide_chunks_with_usage)
    monkeypatch.setattr(retriever, "rerank_guide_chunks", fake_rerank_guide_chunks)
    monkeypatch.setattr(retriever, "get_cached_json", async_fake_get_cached)
    monkeypatch.setattr(retriever, "set_cached_json", async_fake_set_cached)

    results, _, _ = await retriever.retrieve_travel_guide("火星 沙漠 极地科考", top_k=2)

    assert results == []


async def async_fake_get_cached(_key):
    return None


async def async_fake_set_cached(*_args, **_kwargs):
    return None
