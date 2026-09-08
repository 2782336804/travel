from app.config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MAX_RETRIES,
    LLM_MODEL,
    LLM_TIMEOUT_SECONDS,
)


def _build_chat_llm():
    """创建通用 ChatOpenAI 实例。"""
    if not LLM_API_KEY:
        return None

    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        return None

    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.3,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL or None,
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=LLM_MAX_RETRIES,
    )


def _extract_token_usage(response) -> dict[str, int]:
    """从 LangChain AIMessage 中提取 token 使用量。"""
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    metadata = getattr(response, "response_metadata", None) or {}
    token_usage = metadata.get("token_usage", {})
    if token_usage:
        usage["prompt_tokens"] = token_usage.get("prompt_tokens", 0)
        usage["completion_tokens"] = token_usage.get("completion_tokens", 0)
    return usage


async def get_city_by_description(user_query: str) -> tuple[str, dict[str, int]]:
    """把用户描述提炼成唯一城市名。返回 (city, token_usage)。

    LLM 不可用或调用失败时返回原字符串和空 usage，由上层继续降级。
    """
    prompt = f"""
你是城市提取器。
根据用户描述判断对应的唯一城市。
规则：
1. 如果可以确定唯一城市：仅输出城市名称，禁止输出“市”，禁止解释、标点、换行、多余文字。
2. 如果无法确定唯一城市，输出特殊标记：@@UNKNOWN@@

用户描述：{user_query}
输出：
"""
    empty_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    llm = _build_chat_llm()
    if llm is None:
        return user_query, empty_usage

    try:
        response = await llm.ainvoke([{"role": "user", "content": prompt}])
    except Exception as exc:
        print(f"[model_tool] 城市提取调用失败: {type(exc).__name__}: {exc}")
        return user_query, empty_usage

    token_usage = _extract_token_usage(response)
    print(
        "[model_tool] 城市提取调用完成。token: "
        f"prompt={token_usage['prompt_tokens']}, completion={token_usage['completion_tokens']}"
    )

    res = response.content
    if isinstance(res, list):
        res = "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in res
        )
    # 判断模型返回标记：明确无法确定时保留标记交由上层判定；
    # 仅当 LLM 不可用或调用异常时才降级返回原输入。
    if res == "@@UNKNOWN@@":
        return "@@UNKNOWN@@", token_usage
    return res, token_usage
