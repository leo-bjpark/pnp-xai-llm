"""LangGraph booking multi-agent graph."""

from __future__ import annotations

import re
import uuid
from datetime import date
from typing import Any, Literal, Optional

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from typing_extensions import Annotated, TypedDict


class Context(TypedDict, total=False):
    """Context parameters configurable at run time."""

    default_city: str
    default_user_id: str


class State(TypedDict, total=False):
    """Agent state."""

    messages: Annotated[list[AnyMessage], add_messages]
    user_id: str
    bookings: list[dict[str, Any]]
    intent: Literal["search", "book", "cancel", "list", "policy", "fallback"]
    last_result: dict[str, Any]


OFFERS: list[dict[str, Any]] = [
    {
        "offer_id": "OFR-1001",
        "city": "서울",
        "hotel_name": "Han River Hotel",
        "room_type": "Standard Double",
        "nightly_price": 95000,
        "max_guests": 2,
        "breakfast": True,
    },
    {
        "offer_id": "OFR-2001",
        "city": "서울",
        "hotel_name": "Namsan Stay",
        "room_type": "Deluxe Twin",
        "nightly_price": 125000,
        "max_guests": 3,
        "breakfast": False,
    },
    {
        "offer_id": "OFR-3001",
        "city": "부산",
        "hotel_name": "Haeundae Breeze",
        "room_type": "Ocean View",
        "nightly_price": 140000,
        "max_guests": 2,
        "breakfast": True,
    },
]

POLICY_TEXT = {
    "cancel": "체크인 24시간 전까지 무료 취소 가능, 이후 취소 수수료가 발생할 수 있습니다.",
    "refund": "환불은 취소 시점 정책에 따라 영업일 3~5일 내 처리됩니다.",
    "change": "일정 변경은 객실 가용성과 요금 차액에 따라 가능 여부가 결정됩니다.",
}


def _latest_human_text(messages: list[AnyMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = message.content
            if isinstance(content, str):
                return content
            return str(content)
    return ""


def _normalize_question_text(text: str) -> str:
    normalized = text.strip()
    match = re.match(r"^\s*(question|질문)\s*:\s*(.+)$", normalized, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(2).strip()
    return normalized


def _extract_guests(text: str) -> int:
    match = re.search(r"(\d+)\s*명", text)
    return int(match.group(1)) if match else 2


def _extract_budget(text: str) -> Optional[int]:
    match_manwon = re.search(r"(\d+)\s*만\s*원?", text)
    if match_manwon:
        return int(match_manwon.group(1)) * 10000
    match_won = re.search(r"(\d{4,})\s*원", text)
    if match_won:
        return int(match_won.group(1))
    return None


def _extract_city(text: str, default_city: str) -> str:
    for city in ("서울", "부산", "제주"):
        if city in text:
            return city
    return default_city


def _extract_stay_nights(text: str) -> int:
    match = re.search(r"(\d{4}-\d{2}-\d{2})\s*[~\-]\s*(\d{4}-\d{2}-\d{2})", text)
    if not match:
        return 1
    try:
        checkin = date.fromisoformat(match.group(1))
        checkout = date.fromisoformat(match.group(2))
        nights = (checkout - checkin).days
        return max(nights, 1)
    except ValueError:
        return 1


def _filter_offers(text: str, default_city: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    guests = _extract_guests(text)
    budget = _extract_budget(text)
    nights = _extract_stay_nights(text)
    city = _extract_city(text, default_city)
    needs_breakfast = "조식" in text

    filtered: list[dict[str, Any]] = []
    for offer in OFFERS:
        if offer["city"] != city:
            continue
        if offer["max_guests"] < guests:
            continue
        if needs_breakfast and not offer["breakfast"]:
            continue
        total_price = offer["nightly_price"] * nights
        if budget is not None and total_price > budget:
            continue
        filtered.append({**offer, "nights": nights, "total_price": total_price})

    filtered.sort(key=lambda x: x["total_price"])
    criteria = {
        "city": city,
        "guests": guests,
        "budget": budget,
        "nights": nights,
        "breakfast": needs_breakfast,
    }
    return filtered, criteria


def _resolve_user_and_context(
    state: State, runtime: Runtime[Context]
) -> tuple[str, str, list[dict[str, Any]], str]:
    context = runtime.context or {}
    user_id = state.get("user_id") or context.get("default_user_id") or "guest"
    default_city = context.get("default_city") or "서울"
    bookings = list(state.get("bookings", []))
    text = _normalize_question_text(_latest_human_text(state.get("messages", [])))
    return user_id, default_city, bookings, text


def _route_intent(state: State) -> str:
    return state.get("intent", "fallback")


async def intent_router(state: State, runtime: Runtime[Context]) -> dict[str, Any]:
    """Classify request intent and normalize base state."""
    user_id, _default_city, bookings, text = _resolve_user_and_context(state, runtime)

    if not text:
        return {
            "intent": "fallback",
            "user_id": user_id,
            "bookings": bookings,
            "last_result": {"action": "fallback", "reason": "empty_input"},
        }

    if "정책" in text or "환불" in text:
        intent: Literal["search", "book", "cancel", "list", "policy", "fallback"] = "policy"
    elif "취소" in text:
        intent = "cancel"
    elif "목록" in text or "내 예약" in text:
        intent = "list"
    elif "예약" in text:
        intent = "book"
    elif any(token in text for token in ("가능", "상품", "추천", "찾아", "조회")):
        intent = "search"
    else:
        intent = "fallback"

    return {"intent": intent, "user_id": user_id, "bookings": bookings}


async def policy_agent(state: State, runtime: Runtime[Context]) -> dict[str, Any]:
    """Answer policy questions."""
    user_id, _default_city, bookings, text = _resolve_user_and_context(state, runtime)
    topic = "cancel" if "취소" in text else "refund" if "환불" in text else "change"
    answer = f"정책 안내: {POLICY_TEXT[topic]}"
    return {
        "messages": [AIMessage(content=answer)],
        "user_id": user_id,
        "bookings": bookings,
        "last_result": {"action": "policy", "topic": topic},
    }


async def search_agent(state: State, runtime: Runtime[Context]) -> dict[str, Any]:
    """Find matching offers without creating a booking."""
    user_id, default_city, bookings, text = _resolve_user_and_context(state, runtime)
    offers, criteria = _filter_offers(text, default_city)

    if not offers:
        answer = (
            "조건에 맞는 상품이 없습니다. "
            f"(도시={criteria['city']}, 인원={criteria['guests']}, 예산={criteria['budget']})"
        )
        return {
            "messages": [AIMessage(content=answer)],
            "user_id": user_id,
            "bookings": bookings,
            "last_result": {"action": "search", "offers": []},
        }

    preview_lines = []
    for idx, offer in enumerate(offers[:3], start=1):
        reason = "가성비 우수" if idx == 1 else "조건 적합"
        preview_lines.append(
            f"{idx}. {offer['hotel_name']} ({offer['room_type']})\n"
            f"   - 총액: {offer['total_price']}원 / {offer['nights']}박\n"
            f"   - 추천 이유: {reason}"
        )
    answer = (
        "여행 추천 결과입니다. 에어비앤비처럼 조건 기반으로 골라봤어요.\n"
        + "\n".join(preview_lines)
    )
    return {
        "messages": [AIMessage(content=answer)],
        "user_id": user_id,
        "bookings": bookings,
        "last_result": {"action": "search", "offers": offers[:3]},
    }


async def book_agent(state: State, runtime: Runtime[Context]) -> dict[str, Any]:
    """Create a booking from best matched offer."""
    user_id, default_city, bookings, text = _resolve_user_and_context(state, runtime)
    offers, criteria = _filter_offers(text, default_city)
    if not offers:
        answer = (
            "예약 가능한 상품이 없습니다. "
            f"(도시={criteria['city']}, 인원={criteria['guests']}, 예산={criteria['budget']})"
        )
        return {
            "messages": [AIMessage(content=answer)],
            "user_id": user_id,
            "bookings": bookings,
            "last_result": {"action": "book", "status": "failed"},
        }

    if not text:
        answer = "요청 메시지가 없습니다. 원하는 예약 조건을 입력해 주세요."
        return {"messages": [AIMessage(content=answer)], "user_id": user_id, "bookings": bookings}

    chosen = offers[0]
    booking = {
        "booking_id": f"BKG-{uuid.uuid4().hex[:8]}",
        "user_id": user_id,
        "offer_id": chosen["offer_id"],
        "hotel_name": chosen["hotel_name"],
        "room_type": chosen["room_type"],
        "city": chosen["city"],
        "nights": chosen["nights"],
        "total_price": chosen["total_price"],
        "status": "confirmed",
    }
    bookings.append(booking)
    answer = (
        f"예약 완료: {booking['hotel_name']} {booking['room_type']} / "
        f"{booking['nights']}박 / {booking['total_price']}원 / ID={booking['booking_id']}"
    )
    return {
        "messages": [AIMessage(content=answer)],
        "user_id": user_id,
        "bookings": bookings,
        "last_result": {"action": "book", "booking": booking},
    }


async def cancel_agent(state: State, runtime: Runtime[Context]) -> dict[str, Any]:
    """Cancel latest confirmed booking."""
    user_id, _default_city, bookings, _text = _resolve_user_and_context(state, runtime)
    confirmed = [b for b in bookings if b.get("status") == "confirmed"]
    if not confirmed:
        answer = "취소할 예약이 없습니다."
        return {
            "messages": [AIMessage(content=answer)],
            "user_id": user_id,
            "bookings": bookings,
            "last_result": {"action": "cancel", "status": "failed"},
        }

    target = confirmed[-1]
    target["status"] = "canceled"
    answer = f"예약 취소 완료: {target['booking_id']} ({target['hotel_name']})"
    return {
        "messages": [AIMessage(content=answer)],
        "user_id": user_id,
        "bookings": bookings,
        "last_result": {"action": "cancel", "booking": target},
    }


async def list_agent(state: State, runtime: Runtime[Context]) -> dict[str, Any]:
    """List all bookings for current user in state."""
    user_id, _default_city, bookings, _text = _resolve_user_and_context(state, runtime)
    if not bookings:
        answer = "현재 예약 내역이 없습니다."
    else:
        lines = [
            f"- {b['booking_id']} | {b['hotel_name']} | {b['status']} | {b['total_price']}원"
            for b in bookings
        ]
        answer = "예약 목록:\n" + "\n".join(lines)
    return {"messages": [AIMessage(content=answer)], "user_id": user_id, "bookings": bookings}


async def fallback_agent(state: State, runtime: Runtime[Context]) -> dict[str, Any]:
    """Handle unknown intent."""
    user_id, _default_city, bookings, _text = _resolve_user_and_context(state, runtime)
    answer = (
        "요청 형식을 다시 입력해 주세요.\n"
        "예시: Question: 서울에서 2026-03-01~2026-03-03 2명, 조식 포함, 예산 20만원으로 추천해줘"
    )
    return {
        "messages": [AIMessage(content=answer)],
        "user_id": user_id,
        "bookings": bookings,
        "last_result": {"action": "fallback"},
    }


graph = (
    StateGraph(State, context_schema=Context)
    .add_node("intent_router", intent_router)
    .add_node("policy_agent", policy_agent)
    .add_node("search_agent", search_agent)
    .add_node("book_agent", book_agent)
    .add_node("cancel_agent", cancel_agent)
    .add_node("list_agent", list_agent)
    .add_node("fallback_agent", fallback_agent)
    .add_edge(START, "intent_router")
    .add_conditional_edges(
        "intent_router",
        _route_intent,
        {
            "policy": "policy_agent",
            "search": "search_agent",
            "book": "book_agent",
            "cancel": "cancel_agent",
            "list": "list_agent",
            "fallback": "fallback_agent",
        },
    )
    .add_edge("policy_agent", END)
    .add_edge("search_agent", END)
    .add_edge("book_agent", END)
    .add_edge("cancel_agent", END)
    .add_edge("list_agent", END)
    .add_edge("fallback_agent", END)
    .compile(name="Booking Agent Graph")
)
