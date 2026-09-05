from pathlib import Path
import sys

import pytest


CURRENT_FILE = Path(__file__).resolve()
BACKEND_DIR = CURRENT_FILE.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import app.services.map_service as map_service  # noqa: E402
import app.services.place_candidate_service as candidate_service  # noqa: E402
from app.services.place_candidate_service import PlaceCandidateCategory  # noqa: E402


def build_place(
    poi_id: str,
    name: str,
    cityname: str = "上海市",
    adcode: str = "310101",
    latitude: float | None = 31.23,
    longitude: float | None = 121.47,
) -> dict[str, object]:
    return {
        "poi_id": poi_id,
        "name": name,
        "address": "测试地址",
        "cityname": cityname,
        "adname": "黄浦区",
        "adcode": adcode,
        "type": "测试类型",
        "latitude": latitude,
        "longitude": longitude,
        "image_url": None,
    }


async def test_collect_city_candidate_pool_queries_three_strict_categories(monkeypatch) -> None:
    """候选池应按行政区严格查询景点、餐饮和住宿三类 POI。"""
    captured_calls: list[dict[str, object]] = []

    async def fake_search_places(**kwargs):
        captured_calls.append(kwargs)
        return [build_place(f"id-{len(captured_calls)}", str(kwargs["keyword"]))]

    monkeypatch.setattr(candidate_service, "search_places", fake_search_places)

    pool = await candidate_service.collect_city_candidate_pool(
        city="上海",
        adcode="310000",
        minimum_counts={
            PlaceCandidateCategory.SPOT: 1,
            PlaceCandidateCategory.MEAL: 1,
            PlaceCandidateCategory.HOTEL: 1,
        },
    )

    assert captured_calls == [
        {
            "keyword": "景点",
            "city": "310000",
            "page_size": 25,
            "types": "风景名胜",
            "city_limit": True,
        },
        {
            "keyword": "美食",
            "city": "310000",
            "page_size": 25,
            "types": "餐饮服务",
            "city_limit": True,
        },
        {
            "keyword": "酒店",
            "city": "310000",
            "page_size": 25,
            "types": "住宿服务",
            "city_limit": True,
        },
    ]
    assert pool.meets_minimum is True


async def test_candidate_pool_filters_cross_city_missing_coordinates_and_duplicates(monkeypatch) -> None:
    """跨城、无坐标和重复 POI 不能进入动态候选池。"""
    raw_places = [
        build_place("valid-1", "上海测试地点"),
        build_place(
            "cross-city",
            "杭州测试地点",
            cityname="杭州市",
            adcode="330106",
        ),
        build_place("missing-location", "无坐标地点", latitude=None),
        build_place("valid-1", "重复地点"),
    ]

    async def fake_search_places(**_kwargs):
        return raw_places

    monkeypatch.setattr(candidate_service, "search_places", fake_search_places)

    pool = await candidate_service.collect_city_candidate_pool(
        city="上海",
        adcode="310000",
        minimum_counts={
            PlaceCandidateCategory.SPOT: 1,
            PlaceCandidateCategory.MEAL: 1,
            PlaceCandidateCategory.HOTEL: 1,
        },
    )

    for category in PlaceCandidateCategory:
        candidates = pool.candidates_for(category)
        assert [candidate.poi_id for candidate in candidates] == ["valid-1"]
    assert pool.meets_minimum is True


async def test_candidate_pool_reports_category_shortages(monkeypatch) -> None:
    """任一类别不足时，候选池必须明确给出缺口。"""

    async def fake_search_places(**kwargs):
        return (
            [build_place("spot-1", "上海景点")]
            if kwargs["types"] == "风景名胜"
            else []
        )

    monkeypatch.setattr(candidate_service, "search_places", fake_search_places)

    pool = await candidate_service.collect_city_candidate_pool(
        city="上海",
        minimum_counts={
            PlaceCandidateCategory.SPOT: 1,
            PlaceCandidateCategory.MEAL: 1,
            PlaceCandidateCategory.HOTEL: 1,
        },
    )

    assert pool.meets_minimum is False
    assert pool.shortages == {
        PlaceCandidateCategory.MEAL: 1,
        PlaceCandidateCategory.HOTEL: 1,
    }


async def test_candidate_pool_accepts_district_pois_by_adcode(monkeypatch) -> None:
    """区县级旅游城市应按 adcode 接受 POI，而不是被上级 cityname 误过滤。"""
    raw_places = [
        build_place(
            "dunhuang-1",
            "敦煌测试地点",
            cityname="酒泉市",
            adcode="620982",
        ),
        build_place(
            "jiuquan-1",
            "酒泉其他区县地点",
            cityname="酒泉市",
            adcode="620902",
        ),
    ]

    async def fake_search_places(**_kwargs):
        return raw_places

    monkeypatch.setattr(candidate_service, "search_places", fake_search_places)

    pool = await candidate_service.collect_city_candidate_pool(
        city="敦煌",
        adcode="620982",
        administrative_level="district",
        minimum_counts={category: 1 for category in PlaceCandidateCategory},
    )

    for category in PlaceCandidateCategory:
        assert [
            candidate.poi_id
            for candidate in pool.candidates_for(category)
        ] == ["dunhuang-1"]
    assert pool.meets_minimum is True


async def test_map_search_places_sends_types_and_city_limit(monkeypatch) -> None:
    """地图适配层必须把分类和城市强限制传给高德 v3 接口。"""
    captured: dict[str, object] = {}
    monkeypatch.setattr(map_service, "get_cached_json", async_fake_get_cached)
    monkeypatch.setattr(map_service, "set_cached_json", async_fake_set_cached)

    async def fake_request(path: str, params: dict[str, object]) -> dict[str, object]:
        captured["path"] = path
        captured["params"] = params
        return {"pois": []}

    monkeypatch.setattr(map_service, "_request_amap", fake_request)

    assert await map_service.search_places(
        keyword="景点",
        city="310000",
        page_size=25,
        types="风景名胜",
        city_limit=True,
    ) == []
    assert captured["path"] == "/place/text"
    assert captured["params"] == {
        "keywords": "景点",
        "city": "310000",
        "offset": 25,
        "page": 1,
        "extensions": "all",
        "types": "风景名胜",
        "citylimit": "true",
    }


async def test_candidate_pool_wraps_map_failure(monkeypatch) -> None:
    """地图故障应转换为候选采集故障，供 API 返回独立 503。"""

    async def fail_search_places(**_kwargs):
        raise RuntimeError("高德服务不可用")

    monkeypatch.setattr(candidate_service, "search_places", fail_search_places)

    with pytest.raises(
        candidate_service.CandidateCollectionUnavailableError,
        match="暂时无法获取“上海”的景点候选",
    ) as exc_info:
        await candidate_service.collect_city_candidate_pool(city="上海")

    assert exc_info.value.reason == "map_service_unavailable"
    assert exc_info.value.category == "spot"


async def test_candidate_pool_preserves_safe_amap_reason(monkeypatch) -> None:
    """候选采集错误应保留脱敏 infocode 和失败类别。"""

    async def fail_search_places(**_kwargs):
        raise map_service.AmapServiceError(
            "高德地图接口调用失败：访问已超出日访问量",
            reason="10003",
        )

    monkeypatch.setattr(candidate_service, "search_places", fail_search_places)

    with pytest.raises(
        candidate_service.CandidateCollectionUnavailableError,
    ) as exc_info:
        await candidate_service.collect_city_candidate_pool(city="杭州")

    assert exc_info.value.reason == "10003"
    assert exc_info.value.category == "spot"


async def test_request_amap_raises_sanitized_business_error(monkeypatch) -> None:
    """高德业务错误应保留 infocode，但错误文本不得包含 API Key。"""

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {
                "status": "0",
                "info": "访问已超出日访问量",
                "infocode": "10003",
            }

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def get(self, *_args, **_kwargs) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(map_service, "AMAP_API_KEY", "sensitive-test-key")
    monkeypatch.setattr(map_service, "_build_client", lambda: FakeClient())

    with pytest.raises(map_service.AmapServiceError) as exc_info:
        await map_service._request_amap("/config/district", {"keywords": "杭州"})

    assert exc_info.value.reason == "10003"
    assert "sensitive-test-key" not in str(exc_info.value)


async def test_request_amap_rejects_non_object_response(monkeypatch) -> None:
    """合法 JSON 的根节点若不是对象，也应归一为可诊断的地图错误。"""

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> list[object]:
            return []

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def get(self, *_args, **_kwargs) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(map_service, "AMAP_API_KEY", "sensitive-test-key")
    monkeypatch.setattr(map_service, "_build_client", lambda: FakeClient())

    with pytest.raises(map_service.AmapServiceError) as exc_info:
        await map_service._request_amap("/config/district", {"keywords": "杭州"})

    assert exc_info.value.reason == "invalid_response"
    assert "sensitive-test-key" not in str(exc_info.value)


async def async_fake_get_cached(_key):
    return None


async def async_fake_set_cached(*_args, **_kwargs):
    return None
