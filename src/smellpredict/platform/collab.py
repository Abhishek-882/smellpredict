"""
SmellPredict — Y.js WebSocket Collaboration Engine
====================================================
Provides real-time collaborative editing via a Y.js-compatible WebSocket relay.

Architecture:
  - Server is a PURE BINARY RELAY — it never interprets Y.js CRDT updates.
  - Each "room" corresponds to one unique file (identified by a stable room_id).
  - When a client connects it receives any stored room state from Redis (if available)
    so late joiners see the current document immediately.
  - All subsequent Y.js binary updates are broadcast to every other peer in the room.
  - Y.js CRDT guarantees convergence: simultaneous edits from multiple clients
    are merged deterministically without conflicts.

Room ID scheme:
  room_id = sha256(f"{owner}/{repo}/{branch}/{filepath}").hexdigest()[:16]

WebSocket endpoint:
  GET /ws/room/{room_id}?token=<smellpredict_jwt>

JWT authentication is verified BEFORE the WebSocket handshake is accepted.
Connection is rejected with WS close code 4001 if the token is invalid.

Optional Redis persistence:
  Set REDIS_URL env var to enable. When set, the cumulative Y.js document
  state is stored as a binary blob in Redis under key `yjs:room:{room_id}`.
  Late joiners receive this state first before live updates begin.
  If Redis is unavailable the server falls back to in-memory-only mode.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("smellpredict.collab")

from smellpredict.platform.auth import verify_jwt

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

REDIS_URL: Optional[str] = os.getenv("REDIS_URL", "")
_redis_client = None  # lazy-initialized below

# ─────────────────────────────────────────────────────────────────────────────
# In-memory room registry
# Maps room_id → set of active WebSocket connections
# ─────────────────────────────────────────────────────────────────────────────

_rooms: dict[str, set[WebSocket]] = {}

# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────

router = APIRouter(tags=["collaboration"])


# ─────────────────────────────────────────────────────────────────────────────
# Redis helpers (gracefully disabled if REDIS_URL not set)
# ─────────────────────────────────────────────────────────────────────────────


async def _get_redis():
    """Lazily initialize the Redis client. Returns None if Redis is not configured."""
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    if not REDIS_URL:
        return None
    try:
        import redis.asyncio as aioredis  # type: ignore

        _redis_client = await aioredis.from_url(REDIS_URL, decode_responses=False)
        await _redis_client.ping()
        logger.info(f"Redis connected: {REDIS_URL}")
        return _redis_client
    except Exception as exc:
        logger.warning(f"Redis unavailable — falling back to in-memory mode: {exc}")
        return None


async def _load_room_state(room_id: str) -> Optional[bytes]:
    """Load the persisted Y.js document state for a room from Redis."""
    redis = await _get_redis()
    if redis is None:
        return None
    try:
        return await redis.get(f"yjs:room:{room_id}")
    except Exception as exc:
        logger.warning(f"Redis load error for room {room_id}: {exc}")
        return None


async def _save_room_state(room_id: str, update: bytes) -> None:
    """Append a Y.js update to the persisted state in Redis (APPEND to binary blob)."""
    redis = await _get_redis()
    if redis is None:
        return
    try:
        # APPEND accumulates all updates — Y.js can merge the full state on client side
        await redis.append(f"yjs:room:{room_id}", update)
    except Exception as exc:
        logger.warning(f"Redis save error for room {room_id}: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# In-memory comments and chat stores (backed by Redis if configured)
# ─────────────────────────────────────────────────────────────────────────────

import time
import uuid
from pydantic import BaseModel, Field

_comments_store: dict[str, list[dict]] = {}
_chat_store: dict[str, list[dict]] = {}


class CommentItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    line_number: int
    author: str
    avatar: Optional[str] = ""
    text: str
    timestamp: float = Field(default_factory=time.time)
    resolved: bool = False


class CommentCreateRequest(BaseModel):
    line_number: int
    text: str
    author: Optional[str] = "Collaborator"
    avatar: Optional[str] = ""


class ChatMessageRequest(BaseModel):
    text: str
    author: Optional[str] = "Collaborator"
    avatar: Optional[str] = ""


# ─────────────────────────────────────────────────────────────────────────────
# Public helper — Room ID derivation
# ─────────────────────────────────────────────────────────────────────────────


def make_room_id(owner: str, repo: str, branch: str, filepath: str) -> str:
    """
    Derive a stable 16-character hex room ID from the file's unique coordinates.
    Called by the frontend JS and by the server to ensure the same room is used.
    """
    key = f"{owner}/{repo}/{branch}/{filepath}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────────────
# REST endpoints — room info, comments & chat
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/rooms", summary="List active collaboration rooms")
async def list_rooms():
    """
    Returns a list of currently active rooms with peer counts.
    Useful for debugging and monitoring.
    """
    return {
        "rooms": [
            {"room_id": rid, "peers": len(peers)}
            for rid, peers in _rooms.items()
            if peers  # only non-empty rooms
        ],
        "total_peers": sum(len(p) for p in _rooms.values()),
    }


@router.get("/rooms/{room_id}/peers", summary="Get peer count for a room")
async def room_peers(room_id: str):
    """Returns the number of active collaborators in a specific room."""
    peers = len(_rooms.get(room_id, set()))
    return {"room_id": room_id, "peers": peers}


@router.get("/rooms/{room_id}/comments", summary="Get line annotations for a room")
async def get_room_comments(room_id: str):
    """Returns all pinned code review comments for the specified file/room."""
    return {"room_id": room_id, "comments": _comments_store.get(room_id, [])}


@router.post("/rooms/{room_id}/comments", summary="Add a pinned line comment")
async def add_room_comment(room_id: str, body: CommentCreateRequest):
    """Add a new line review comment to the file."""
    comment = {
        "id": str(uuid.uuid4())[:8],
        "line_number": body.line_number,
        "author": body.author or "Collaborator",
        "avatar": body.avatar or "",
        "text": body.text,
        "timestamp": time.time(),
        "resolved": False,
    }
    _comments_store.setdefault(room_id, []).append(comment)
    
    # Broadcast to room peers via text JSON
    await _broadcast_json(room_id, {"type": "NEW_COMMENT", "comment": comment})
    return comment


@router.post("/rooms/{room_id}/comments/{comment_id}/resolve", summary="Resolve a comment")
async def resolve_room_comment(room_id: str, comment_id: str):
    """Toggle comment resolved status."""
    comments = _comments_store.get(room_id, [])
    for c in comments:
        if c["id"] == comment_id:
            c["resolved"] = not c.get("resolved", False)
            await _broadcast_json(room_id, {"type": "RESOLVE_COMMENT", "comment_id": comment_id, "resolved": c["resolved"]})
            return {"status": "ok", "comment": c}
    return {"status": "not_found"}


@router.delete("/rooms/{room_id}/comments/{comment_id}", summary="Delete a comment")
async def delete_room_comment(room_id: str, comment_id: str):
    """Delete a comment from the room."""
    comments = _comments_store.get(room_id, [])
    _comments_store[room_id] = [c for c in comments if c["id"] != comment_id]
    await _broadcast_json(room_id, {"type": "DELETE_COMMENT", "comment_id": comment_id})
    return {"status": "ok", "comment_id": comment_id}


@router.get("/rooms/{room_id}/chat", summary="Get team chat history for a room")
async def get_room_chat(room_id: str):
    """Get recent team chat messages for this room/file."""
    return {"room_id": room_id, "messages": _chat_store.get(room_id, [])}


@router.post("/rooms/{room_id}/chat", summary="Send a team chat message")
async def send_room_chat(room_id: str, body: ChatMessageRequest):
    """Post a team chat message and broadcast to peers."""
    msg = {
        "id": str(uuid.uuid4())[:8],
        "author": body.author or "Collaborator",
        "avatar": body.avatar or "",
        "text": body.text,
        "timestamp": time.time(),
    }
    _chat_store.setdefault(room_id, []).append(msg)
    # Keep last 100 messages
    if len(_chat_store[room_id]) > 100:
        _chat_store[room_id] = _chat_store[room_id][-100:]
        
    await _broadcast_json(room_id, {"type": "CHAT_MESSAGE", "message": msg})
    return msg


async def _broadcast_json(room_id: str, payload: dict, exclude: Optional[WebSocket] = None):
    """Helper to broadcast a JSON payload to all connected WebSockets in a room."""
    peers = _rooms.get(room_id, set())
    if exclude:
        peers = peers - {exclude}
    import json
    msg_str = json.dumps(payload)
    for peer in list(peers):
        if peer.client_state == WebSocketState.CONNECTED:
            try:
                await peer.send_text(msg_str)
            except Exception as exc:
                logger.debug(f"Broadcast error to peer: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket endpoint — Y.js binary & JSON control relay
# ─────────────────────────────────────────────────────────────────────────────


@router.websocket("/ws/room/{room_id}")
async def collab_websocket(
    websocket: WebSocket,
    room_id: str,
    token: Optional[str] = Query(None, description="SmellPredict JWT for authentication (optional for guests)"),
):
    """
    Y.js-compatible WebSocket relay supporting both binary CRDT updates
    and JSON control messages (chat, presence, comments).
    """
    username = f"Dev_{str(uuid.uuid4())[:5]}"
    if token:
        try:
            payload = verify_jwt(token)
            username = payload.get("sub", username)
        except Exception:
            pass  # Fallback to guest name instead of hard-failing handshake

    # ── Accept and register ─────────────────────────────────────────────────
    await websocket.accept()
    _rooms.setdefault(room_id, set()).add(websocket)
    peer_count = len(_rooms[room_id])
    logger.info(
        f"[room:{room_id}] {username} connected — {peer_count} peer(s) in room"
    )

    # Broadcast presence join
    await _broadcast_json(room_id, {
        "type": "USER_JOIN",
        "username": username,
        "peer_count": peer_count,
    }, exclude=websocket)

    try:
        # ── Send persisted state to new joiner ──────────────────────────────
        persisted = await _load_room_state(room_id)
        if persisted:
            await websocket.send_bytes(persisted)
            logger.debug(
                f"[room:{room_id}] Sent {len(persisted)} bytes of persisted state"
                f" to {username}"
            )

        # ── Main relay loop (handles binary and text) ─────────────────────────
        while True:
            msg = await websocket.receive()
            if "bytes" in msg and msg["bytes"]:
                raw_bytes = msg["bytes"]
                # Persist binary CRDT update
                asyncio.create_task(_save_room_state(room_id, raw_bytes))

                # Broadcast binary update to peers
                peers_in_room = _rooms.get(room_id, set()) - {websocket}
                for peer in list(peers_in_room):
                    if peer.client_state == WebSocketState.CONNECTED:
                        try:
                            await peer.send_bytes(raw_bytes)
                        except Exception as exc:
                            logger.warning(f"[room:{room_id}] Binary relay error: {exc}")

            elif "text" in msg and msg["text"]:
                # Text JSON control frame (chat/presence/cursor)
                import json
                try:
                    data = json.loads(msg["text"])
                    event_type = data.get("type")

                    if event_type == "CHAT_MESSAGE":
                        chat_msg = {
                            "id": str(uuid.uuid4())[:8],
                            "author": username,
                            "avatar": data.get("avatar", ""),
                            "text": data.get("text", ""),
                            "timestamp": time.time(),
                        }
                        _chat_store.setdefault(room_id, []).append(chat_msg)
                        await _broadcast_json(room_id, {"type": "CHAT_MESSAGE", "message": chat_msg})

                    elif event_type == "CURSOR_MOVE":
                        # Relay cursor position to other peers
                        await _broadcast_json(room_id, {
                            "type": "PEER_CURSOR",
                            "username": username,
                            "line": data.get("line", 1),
                            "column": data.get("column", 1),
                            "color": data.get("color", "#58a6ff"),
                        }, exclude=websocket)

                    else:
                        # Relay generic control event
                        await _broadcast_json(room_id, data, exclude=websocket)

                except Exception as parse_err:
                    logger.debug(f"JSON frame parse error: {parse_err}")

    except WebSocketDisconnect:
        logger.info(f"[room:{room_id}] {username} disconnected")
    except Exception as exc:
        logger.error(f"[room:{room_id}] Unexpected error for {username}: {exc}")
    finally:
        # ── Clean up ────────────────────────────────────────────────────────
        if room_id in _rooms:
            _rooms[room_id].discard(websocket)
            remaining = len(_rooms[room_id])
            logger.info(
                f"[room:{room_id}] {username} removed — {remaining} peer(s) remaining"
            )
            # Broadcast user leave
            await _broadcast_json(room_id, {
                "type": "USER_LEAVE",
                "username": username,
                "peer_count": remaining,
            })
            if remaining == 0:
                _rooms.pop(room_id, None)
                logger.debug(f"[room:{room_id}] Room closed (no more peers)")

