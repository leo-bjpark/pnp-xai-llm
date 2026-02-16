import pytest

from agent import graph

pytestmark = pytest.mark.anyio


@pytest.mark.langsmith
async def test_agent_simple_passthrough() -> None:
    inputs = {"messages": [{"role": "human", "content": "서울 2명 조식 포함으로 예약해줘"}]}
    res = await graph.ainvoke(inputs)
    assert res is not None
    assert "messages" in res
