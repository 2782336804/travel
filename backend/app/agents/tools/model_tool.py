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


async def get_city_by_description(user_query: str) -> str:
    prompt = f"""
你是城市提取器。
根据用户描述判断对应的唯一城市。
规则：
1. 如果可以确定唯一城市：仅输出城市名称，禁止输出“市”，禁止解释、标点、换行、多余文字。
2. 如果无法确定唯一城市，输出特殊标记：@@UNKNOWN@@

用户描述：{user_query}
输出：
"""
    llm = _build_chat_llm()
    if llm is None:
        return user_query
    response = await llm.ainvoke([{"role": "user", "content": prompt}])
    res = response.content
    # 判断模型返回标记，未知就返回原字符串
    if res == "@@UNKNOWN@@":
        return user_query
    return res
