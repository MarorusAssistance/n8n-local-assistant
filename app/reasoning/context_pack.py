from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

from ..config import settings
from ..rag import retrieve_context
from ..token_budget import estimate_text_tokens
from .types import ContextDocChunk, ContextPack, ContextPackBudget, NodeCard, RouterOutput

_NODE_TYPE_RE = re.compile(r"(?im)^Node Type:\s*(.+?)\s*$")
_DISPLAY_NAME_RE = re.compile(r"(?im)^Display Name:\s*(.+?)\s*$")
_REQUIRED_CREDS_RE = re.compile(r"(?im)^Required Credentials:\s*(.+?)\s*$")
_DESCRIPTION_RE = re.compile(r"(?im)^Description:\s*(.+?)\s*$")
_NODE_TYPE_FALLBACK_RE = re.compile(r"\bn8n-nodes-[a-z0-9\-]+\.[a-z0-9A-Z]+\b")


def _compact(text: str, max_chars: int = 320) -> str:
    value = " ".join((text or "").split())
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def _source_from_chunk(chunk: Dict[str, Any]) -> str:
    metadata = chunk.get("metadata")
    metadata_map = metadata if isinstance(metadata, dict) else {}
    linked_kind = str(chunk.get("linked_def_type") or "").strip().lower()
    if str(chunk.get("context_kind") or "").strip().lower() == "linked_def":
        if linked_kind == "node":
            return "nodes"
        if linked_kind == "credential":
            return "credentials"

    kind = str(metadata_map.get("kind") or "").upper()
    if kind.startswith("NODE_") or metadata_map.get("nodeType"):
        return "nodes"
    if kind.startswith("CRED_") or metadata_map.get("credentialType"):
        return "credentials"

    source = str(chunk.get("source") or metadata_map.get("source") or "").strip().lower()
    if source in {
        "nodes",
        "n8n-nodes",
        str(settings.LINKED_DEFS_NODES_SOURCE or "").strip().lower(),
    }:
        return "nodes"
    if source in {
        "credentials",
        "n8n-credentials",
        str(settings.LINKED_DEFS_CREDENTIALS_SOURCE or "").strip().lower(),
    }:
        return "credentials"
    return "docs"


def _first_regex_group(pattern: re.Pattern[str], text: str) -> Optional[str]:
    match = pattern.search(text or "")
    if not match:
        return None
    value = str(match.group(1) or "").strip()
    return value or None


def _extract_node_type(chunk: Dict[str, Any]) -> Optional[str]:
    metadata = chunk.get("metadata")
    metadata_map = metadata if isinstance(metadata, dict) else {}
    node_type = str(metadata_map.get("nodeType") or "").strip()
    if node_type:
        return node_type
    text = str(chunk.get("text") or "")
    from_line = _first_regex_group(_NODE_TYPE_RE, text)
    if from_line:
        return from_line
    fallback_match = _NODE_TYPE_FALLBACK_RE.search(text)
    if fallback_match:
        return fallback_match.group(0)
    linked_entity = str(chunk.get("linked_entity_id") or "").strip()
    return linked_entity or None


def _extract_required_credentials(chunk: Dict[str, Any]) -> List[str]:
    metadata = chunk.get("metadata")
    metadata_map = metadata if isinstance(metadata, dict) else {}
    raw = metadata_map.get("credentialTypes_required")
    if isinstance(raw, list):
        values = [str(item).strip() for item in raw if str(item).strip()]
        if values:
            return values

    text = str(chunk.get("text") or "")
    line = _first_regex_group(_REQUIRED_CREDS_RE, text)
    if not line or line == "-":
        return []
    values = [item.strip() for item in re.split(r"[,\|]", line) if item.strip()]
    return values


def _extract_key_params(chunk: Dict[str, Any]) -> List[str]:
    metadata = chunk.get("metadata")
    metadata_map = metadata if isinstance(metadata, dict) else {}
    raw = metadata_map.get("param_names")
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()][:10]
    return []


def _extract_purpose(chunk: Dict[str, Any]) -> Optional[str]:
    metadata = chunk.get("metadata")
    metadata_map = metadata if isinstance(metadata, dict) else {}
    kind = str(metadata_map.get("kind") or "").upper()
    if kind == "NODE_OVERVIEW":
        return "Node overview and capabilities."
    if kind == "NODE_RESOURCE_OPERATION":
        return "Resource/operation behavior for this node."
    if kind == "NODE_PARAMS_FOR_RESOURCE_OPERATION":
        return "Parameters needed for a specific operation."
    if kind == "NODE_GLOBAL_PARAMS":
        return "Global parameters for the node."
    description = _first_regex_group(_DESCRIPTION_RE, str(chunk.get("text") or ""))
    if description:
        return _compact(description, max_chars=180)
    return None


def _extract_display_name(chunk: Dict[str, Any]) -> Optional[str]:
    metadata = chunk.get("metadata")
    metadata_map = metadata if isinstance(metadata, dict) else {}
    display_name = str(metadata_map.get("displayName") or "").strip()
    if display_name:
        return display_name
    text = str(chunk.get("text") or "")
    return _first_regex_group(_DISPLAY_NAME_RE, text)


def _merge_unique(items: Sequence[str], extra: Sequence[str], limit: int = 12) -> List[str]:
    output: List[str] = []
    seen = set()
    for value in list(items) + list(extra):
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
        if len(output) >= limit:
            break
    return output


def _build_node_cards(chunks: Sequence[Dict[str, Any]], max_node_cards: int) -> List[NodeCard]:
    by_type: Dict[str, NodeCard] = {}
    order: List[str] = []

    for chunk in chunks:
        if _source_from_chunk(chunk) != "nodes":
            continue
        node_type = _extract_node_type(chunk)
        if not node_type:
            continue

        card = by_type.get(node_type)
        if card is None:
            card = NodeCard(type=node_type)
            by_type[node_type] = card
            order.append(node_type)

        display_name = _extract_display_name(chunk)
        purpose = _extract_purpose(chunk)
        required_credentials = _extract_required_credentials(chunk)
        key_params = _extract_key_params(chunk)

        if display_name and not card.displayName:
            card.displayName = display_name
        if purpose and not card.purpose:
            card.purpose = purpose
        card.requiredCredentials = _merge_unique(card.requiredCredentials, required_credentials)
        card.keyParams = _merge_unique(card.keyParams, key_params, limit=16)

    return [by_type[node_type] for node_type in order[:max_node_cards]]


def _build_doc_chunks(chunks: Sequence[Dict[str, Any]], max_doc_chunks: int) -> List[ContextDocChunk]:
    selected: List[ContextDocChunk] = []
    for chunk in chunks:
        text = str(chunk.get("text") or "").strip()
        if not text:
            continue
        source = _source_from_chunk(chunk)
        selected.append(
            ContextDocChunk(
                id=str(chunk.get("doc_id") or f"chunk-{len(selected) + 1}"),
                title=str(chunk.get("title") or "").strip() or None,
                text=text,
                source=source,  # type: ignore[arg-type]
            )
        )
        if len(selected) >= max_doc_chunks:
            break
    return selected


def _context_payload(node_cards: Sequence[NodeCard], doc_chunks: Sequence[ContextDocChunk]) -> str:
    payload = {
        "nodeCards": [card.model_dump(exclude_none=True) for card in node_cards],
        "docChunks": [chunk.model_dump(exclude_none=True) for chunk in doc_chunks],
    }
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _estimated_tokens(node_cards: Sequence[NodeCard], doc_chunks: Sequence[ContextDocChunk]) -> int:
    return estimate_text_tokens(_context_payload(node_cards, doc_chunks))


def _shrink_longest_doc_text(doc_chunks: List[ContextDocChunk]) -> bool:
    if not doc_chunks:
        return False
    max_index = max(range(len(doc_chunks)), key=lambda idx: len(doc_chunks[idx].text))
    current = doc_chunks[max_index].text
    if len(current) <= 160:
        return False
    new_size = max(120, int(len(current) * 0.8))
    if new_size >= len(current):
        return False
    doc_chunks[max_index].text = current[:new_size].rstrip()
    return True


def _trim_to_token_budget(
    node_cards: List[NodeCard],
    doc_chunks: List[ContextDocChunk],
    max_context_tokens: int,
) -> int:
    estimated = _estimated_tokens(node_cards, doc_chunks)
    while estimated > max_context_tokens:
        if _shrink_longest_doc_text(doc_chunks):
            estimated = _estimated_tokens(node_cards, doc_chunks)
            continue
        if doc_chunks:
            doc_chunks.pop()
            estimated = _estimated_tokens(node_cards, doc_chunks)
            continue
        if node_cards:
            node_cards.pop()
            estimated = _estimated_tokens(node_cards, doc_chunks)
            continue
        break
    return estimated


def build_context_pack(
    router_output: RouterOutput,
    user_prompt: str,
    request_id: Optional[str] = None,
) -> ContextPack:
    _ = router_output
    max_node_cards = max(1, settings.MAX_NODE_CARDS)
    max_doc_chunks = max(1, settings.MAX_DOC_CHUNKS)
    max_context_tokens = max(1, settings.MAX_CONTEXT_TOKENS)
    retrieval_top_k = max(12, max_node_cards + max_doc_chunks + 4)

    chunks = retrieve_context(
        user_prompt,
        top_k=retrieval_top_k,
        request_id=request_id,
    )
    node_cards = _build_node_cards(chunks, max_node_cards=max_node_cards)
    doc_chunks = _build_doc_chunks(chunks, max_doc_chunks=max_doc_chunks)
    estimated_tokens = _trim_to_token_budget(
        node_cards=node_cards,
        doc_chunks=doc_chunks,
        max_context_tokens=max_context_tokens,
    )

    budget = ContextPackBudget(
        maxNodeCards=max_node_cards,
        maxDocChunks=max_doc_chunks,
        maxContextTokens=max_context_tokens,
        estimatedTokens=estimated_tokens,
    )
    return ContextPack(nodeCards=node_cards, docChunks=doc_chunks, budget=budget)
