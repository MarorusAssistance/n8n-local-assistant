from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from collections import defaultdict

import re
from urllib.parse import unquote, urlparse


METHOD_URL_EXACT = "url_exact"
METHOD_DOC_KEY_MATCH = "doc_key_match"
METHOD_DERIVED_FROM_NODE = "derived_from_node"
METHOD_MENTION = "mention"
METHOD_SIMILARITY = "similarity"

LINK_METHODS = (
    METHOD_URL_EXACT,
    METHOD_DOC_KEY_MATCH,
    METHOD_DERIVED_FROM_NODE,
    METHOD_MENTION,
    METHOD_SIMILARITY,
)

METHOD_PRIORITY: Dict[str, int] = {
    METHOD_URL_EXACT: 500,
    METHOD_DOC_KEY_MATCH: 450,
    METHOD_DERIVED_FROM_NODE: 350,
    METHOD_MENTION: 250,
    METHOD_SIMILARITY: 150,
}

STOP_DISPLAY_NAMES = {
    "api",
    "node",
    "trigger",
    "action",
    "event",
    "data",
    "file",
    "item",
    "items",
    "date",
    "time",
    "list",
}

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9._-]{1,}")
_MULTISPACE_RE = re.compile(r"\s+")


@dataclass
class DocPage:
    key: str
    text: str
    refs: Set[str] = field(default_factory=set)
    source_field: str = ""
    title: str = ""


@dataclass
class NodeCatalogItem:
    node_type: str
    display_name: str
    doc_refs: Set[str] = field(default_factory=set)
    credential_types_required: List[str] = field(default_factory=list)
    token_set: Set[str] = field(default_factory=set)


@dataclass
class CredentialCatalogItem:
    credential_type: str
    display_name: str
    documentation_key: str
    doc_key_variants: Set[str] = field(default_factory=set)
    supported_nodes: List[str] = field(default_factory=list)
    token_set: Set[str] = field(default_factory=set)


@dataclass(frozen=True)
class NodeLinkCandidate:
    doc_page_key: str
    node_type: str
    confidence: float
    method: str
    evidence: str


@dataclass(frozen=True)
class CredentialLinkCandidate:
    doc_page_key: str
    credential_type: str
    confidence: float
    method: str
    evidence: str


def normalize_reference(value: Any) -> str:
    raw = _as_text(value)
    if not raw:
        return ""

    if "://" in raw or raw.startswith("//"):
        parse_target = raw if "://" in raw else f"https:{raw}"
        parsed = urlparse(parse_target)
        host = (parsed.netloc or "").lower().strip()
        if "@" in host:
            host = host.split("@", 1)[-1]
        if host.endswith(":80") and parsed.scheme.lower() == "http":
            host = host[:-3]
        if host.endswith(":443") and parsed.scheme.lower() == "https":
            host = host[:-4]

        path = _normalize_path_component(parsed.path)
        if host:
            return f"{host}{path}"
        return path

    cleaned = raw.split("#", 1)[0].split("?", 1)[0].strip()
    cleaned = cleaned.replace("\\", "/")
    if not cleaned:
        return ""

    if _looks_like_host_path(cleaned):
        host, path = _split_host_path(cleaned)
        path = _normalize_path_component(path)
        return host.lower() + path

    if _looks_like_page_id(cleaned):
        return cleaned.lower()

    return _normalize_path_component(cleaned)


def normalized_reference_variants(value: Any) -> Set[str]:
    base = normalize_reference(value)
    if not base:
        return set()
    variants = {base}
    host, path = _extract_host_path(base)
    if host and path:
        variants.add(path)
    return variants


def derive_doc_page_key(
    metadata: Mapping[str, Any],
    row_url: Optional[str] = None,
) -> Tuple[str, Set[str], str]:
    selected_field = ""
    selected_value = ""
    refs: Set[str] = set()

    fields_priority = ("page_id", "source_url", "path")
    fallback_fields = ("url", "sourceUrl", "relPath")

    for field in fields_priority:
        value = _as_text(metadata.get(field))
        if value:
            selected_field = field
            selected_value = value
            break

    if not selected_value:
        for field in fallback_fields:
            value = _as_text(metadata.get(field))
            if value:
                selected_field = field
                selected_value = value
                break

    if not selected_value and row_url:
        selected_field = "row_url"
        selected_value = _as_text(row_url)

    for key in list(fields_priority) + list(fallback_fields):
        refs.update(normalized_reference_variants(metadata.get(key)))
    if row_url:
        refs.update(normalized_reference_variants(row_url))

    if not selected_value:
        return "", refs, ""

    if selected_field == "page_id" and _looks_like_page_id(selected_value):
        page_key = selected_value.lower().strip()
    else:
        page_key = normalize_reference(selected_value)
    if not page_key:
        return "", refs, selected_field
    refs.add(page_key)
    return page_key, refs, selected_field


def aggregate_doc_pages(
    rows: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, DocPage], List[str]]:
    pages: Dict[str, DocPage] = {}
    warnings: List[str] = []

    for row in rows:
        metadata_raw = row.get("metadata")
        metadata = metadata_raw if isinstance(metadata_raw, Mapping) else {}
        text = _as_text(row.get("text"))
        row_url = _as_text(row.get("url"))

        page_key, refs, source_field = derive_doc_page_key(metadata, row_url=row_url)
        if not page_key:
            warnings.append("doc chunk skipped: missing doc_page_key candidates (page_id/source_url/path)")
            continue

        title = _as_text(metadata.get("title") or row.get("title"))
        if page_key not in pages:
            pages[page_key] = DocPage(
                key=page_key,
                text=text,
                refs=set(refs),
                source_field=source_field,
                title=title,
            )
            continue

        page = pages[page_key]
        if text:
            page.text = f"{page.text}\n\n{text}".strip()
        page.refs.update(refs)
        if not page.title and title:
            page.title = title

    return pages, warnings


def build_node_catalog(rows: Sequence[Mapping[str, Any]]) -> Dict[str, NodeCatalogItem]:
    catalog: Dict[str, NodeCatalogItem] = {}
    for row in rows:
        metadata_raw = row.get("metadata")
        metadata = metadata_raw if isinstance(metadata_raw, Mapping) else {}
        node_type = _as_text(metadata.get("nodeType"))
        if not node_type:
            continue

        item = catalog.get(node_type)
        if item is None:
            item = NodeCatalogItem(node_type=node_type, display_name=_as_text(metadata.get("displayName")))
            catalog[node_type] = item

        display_name = _as_text(metadata.get("displayName"))
        if display_name and not item.display_name:
            item.display_name = display_name

        doc_urls = metadata.get("doc_urls")
        if isinstance(doc_urls, list):
            for url in doc_urls:
                item.doc_refs.update(normalized_reference_variants(url))
        elif doc_urls:
            item.doc_refs.update(normalized_reference_variants(doc_urls))

        required_creds = metadata.get("credentialTypes_required")
        if isinstance(required_creds, list):
            for cred in required_creds:
                cred_type = _as_text(cred)
                if cred_type and cred_type not in item.credential_types_required:
                    item.credential_types_required.append(cred_type)

    for item in catalog.values():
        item.token_set = _catalog_tokens(item.node_type, item.display_name)
    return catalog


def build_credential_catalog(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, CredentialCatalogItem]:
    catalog: Dict[str, CredentialCatalogItem] = {}
    for row in rows:
        metadata_raw = row.get("metadata")
        metadata = metadata_raw if isinstance(metadata_raw, Mapping) else {}
        credential_type = _as_text(metadata.get("credentialType"))
        if not credential_type:
            continue

        item = catalog.get(credential_type)
        if item is None:
            item = CredentialCatalogItem(
                credential_type=credential_type,
                display_name=_as_text(metadata.get("displayName")),
                documentation_key=_as_text(metadata.get("documentationKey")),
            )
            catalog[credential_type] = item

        display_name = _as_text(metadata.get("displayName"))
        if display_name and not item.display_name:
            item.display_name = display_name

        documentation_key = _as_text(metadata.get("documentationKey"))
        if documentation_key and not item.documentation_key:
            item.documentation_key = documentation_key

        supported_nodes = metadata.get("supportedNodes")
        if isinstance(supported_nodes, list):
            for node in supported_nodes:
                node_type = _as_text(node)
                if node_type and node_type not in item.supported_nodes:
                    item.supported_nodes.append(node_type)

    for item in catalog.values():
        item.doc_key_variants = _documentation_key_variants(item.documentation_key)
        item.token_set = _catalog_tokens(item.credential_type, item.display_name)
    return catalog


def generate_node_links(
    pages: Mapping[str, DocPage],
    nodes: Mapping[str, NodeCatalogItem],
    *,
    mention_limit_per_page: int = 2,
    similarity_limit_per_page: int = 2,
    similarity_threshold: float = 0.55,
) -> List[NodeLinkCandidate]:
    best: Dict[Tuple[str, str], NodeLinkCandidate] = {}
    page_to_nodes: Dict[str, Set[str]] = defaultdict(set)

    for page in pages.values():
        for node in nodes.values():
            intersection = page.refs.intersection(node.doc_refs)
            if not intersection:
                continue
            evidence_ref = sorted(intersection)[0]
            candidate = NodeLinkCandidate(
                doc_page_key=page.key,
                node_type=node.node_type,
                confidence=1.0,
                method=METHOD_URL_EXACT,
                evidence=f"matched_url={evidence_ref}",
            )
            _store_best_node(best, page_to_nodes, candidate)

    for page in pages.values():
        if page_to_nodes.get(page.key):
            continue
        mention_candidates = _find_node_mentions(page, nodes)
        added = 0
        for candidate in mention_candidates:
            if added >= max(0, mention_limit_per_page):
                break
            if _store_best_node(best, page_to_nodes, candidate):
                added += 1

    for page in pages.values():
        if page_to_nodes.get(page.key):
            continue
        similarity_candidates = _find_node_similarity(
            page,
            nodes,
            top_k=max(0, similarity_limit_per_page),
            threshold=similarity_threshold,
        )
        for candidate in similarity_candidates:
            _store_best_node(best, page_to_nodes, candidate)

    return sorted(best.values(), key=lambda item: (item.doc_page_key, item.node_type))


def generate_credential_links(
    pages: Mapping[str, DocPage],
    credentials: Mapping[str, CredentialCatalogItem],
    nodes: Mapping[str, NodeCatalogItem],
    node_links: Sequence[NodeLinkCandidate],
    *,
    mention_limit_per_page: int = 2,
    similarity_limit_per_page: int = 2,
    similarity_threshold: float = 0.55,
) -> Tuple[List[CredentialLinkCandidate], List[str]]:
    best: Dict[Tuple[str, str], CredentialLinkCandidate] = {}
    page_to_credentials: Dict[str, Set[str]] = defaultdict(set)
    warnings: List[str] = []

    for page in pages.values():
        for credential in credentials.values():
            matched_variant = _match_doc_key(page.refs, credential.doc_key_variants)
            if not matched_variant:
                continue
            candidate = CredentialLinkCandidate(
                doc_page_key=page.key,
                credential_type=credential.credential_type,
                confidence=0.95,
                method=METHOD_DOC_KEY_MATCH,
                evidence=f"doc_key={matched_variant}",
            )
            _store_best_credential(best, page_to_credentials, candidate)

    node_links_by_page: Dict[str, List[NodeLinkCandidate]] = defaultdict(list)
    for link in node_links:
        node_links_by_page[link.doc_page_key].append(link)

    for page_key, page_node_links in node_links_by_page.items():
        for node_link in page_node_links:
            node = nodes.get(node_link.node_type)
            if node is None:
                continue
            for credential_type in node.credential_types_required:
                credential = credentials.get(credential_type)
                if credential is None:
                    continue
                confidence = _derived_confidence(node_link.confidence)
                evidence = f"{node_link.node_type} -> {credential_type}"
                if credential.supported_nodes and node_link.node_type not in credential.supported_nodes:
                    confidence = max(0.2, confidence - 0.2)
                    warnings.append(
                        "derived link sanity check: "
                        f"credential {credential_type} does not list node {node_link.node_type} in supportedNodes"
                    )
                candidate = CredentialLinkCandidate(
                    doc_page_key=page_key,
                    credential_type=credential_type,
                    confidence=confidence,
                    method=METHOD_DERIVED_FROM_NODE,
                    evidence=evidence,
                )
                _store_best_credential(best, page_to_credentials, candidate)

    for page in pages.values():
        if page_to_credentials.get(page.key):
            continue
        mention_candidates = _find_credential_mentions(page, credentials)
        added = 0
        for candidate in mention_candidates:
            if added >= max(0, mention_limit_per_page):
                break
            if _store_best_credential(best, page_to_credentials, candidate):
                added += 1

    for page in pages.values():
        if page_to_credentials.get(page.key):
            continue
        similarity_candidates = _find_credential_similarity(
            page,
            credentials,
            top_k=max(0, similarity_limit_per_page),
            threshold=similarity_threshold,
        )
        for candidate in similarity_candidates:
            _store_best_credential(best, page_to_credentials, candidate)

    links = sorted(best.values(), key=lambda item: (item.doc_page_key, item.credential_type))
    return links, warnings


def count_links_by_method(
    node_links: Sequence[NodeLinkCandidate],
    credential_links: Sequence[CredentialLinkCandidate],
) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for link in list(node_links) + list(credential_links):
        counts[link.method] += 1
    return dict(counts)


def _store_best_node(
    best: Dict[Tuple[str, str], NodeLinkCandidate],
    page_to_nodes: Dict[str, Set[str]],
    candidate: NodeLinkCandidate,
) -> bool:
    key = (candidate.doc_page_key, candidate.node_type)
    existing = best.get(key)
    if existing is None or _should_replace(existing.method, existing.confidence, candidate.method, candidate.confidence):
        best[key] = candidate
        page_to_nodes[candidate.doc_page_key].add(candidate.node_type)
        return True
    return False


def _store_best_credential(
    best: Dict[Tuple[str, str], CredentialLinkCandidate],
    page_to_credentials: Dict[str, Set[str]],
    candidate: CredentialLinkCandidate,
) -> bool:
    key = (candidate.doc_page_key, candidate.credential_type)
    existing = best.get(key)
    if existing is None or _should_replace(existing.method, existing.confidence, candidate.method, candidate.confidence):
        best[key] = candidate
        page_to_credentials[candidate.doc_page_key].add(candidate.credential_type)
        return True
    return False


def _should_replace(
    existing_method: str,
    existing_confidence: float,
    new_method: str,
    new_confidence: float,
) -> bool:
    old_priority = METHOD_PRIORITY.get(existing_method, 0)
    new_priority = METHOD_PRIORITY.get(new_method, 0)
    if new_priority > old_priority:
        return True
    if new_priority < old_priority:
        return False
    return new_confidence > existing_confidence


def _find_node_mentions(
    page: DocPage,
    nodes: Mapping[str, NodeCatalogItem],
) -> List[NodeLinkCandidate]:
    text_lower = page.text.lower()
    candidates: List[NodeLinkCandidate] = []
    for node in nodes.values():
        literal = node.node_type.lower()
        if literal and literal in text_lower:
            candidates.append(
                NodeLinkCandidate(
                    doc_page_key=page.key,
                    node_type=node.node_type,
                    confidence=0.72,
                    method=METHOD_MENTION,
                    evidence=f"mention={node.node_type}",
                )
            )
            continue
        display_match = _display_name_mention(node.display_name, text_lower)
        if not display_match:
            continue
        evidence = f"mention={display_match}"
        if page.title:
            evidence += f" title={page.title}"
        candidates.append(
            NodeLinkCandidate(
                doc_page_key=page.key,
                node_type=node.node_type,
                confidence=0.62,
                method=METHOD_MENTION,
                evidence=evidence,
            )
        )
    candidates.sort(key=lambda item: item.confidence, reverse=True)
    return candidates


def _find_credential_mentions(
    page: DocPage,
    credentials: Mapping[str, CredentialCatalogItem],
) -> List[CredentialLinkCandidate]:
    text_lower = page.text.lower()
    candidates: List[CredentialLinkCandidate] = []
    for credential in credentials.values():
        literal = credential.credential_type.lower()
        if literal and literal in text_lower:
            candidates.append(
                CredentialLinkCandidate(
                    doc_page_key=page.key,
                    credential_type=credential.credential_type,
                    confidence=0.68,
                    method=METHOD_MENTION,
                    evidence=f"mention={credential.credential_type}",
                )
            )
            continue
        display_match = _display_name_mention(credential.display_name, text_lower)
        if not display_match:
            continue
        evidence = f"mention={display_match}"
        if page.title:
            evidence += f" title={page.title}"
        candidates.append(
            CredentialLinkCandidate(
                doc_page_key=page.key,
                credential_type=credential.credential_type,
                confidence=0.58,
                method=METHOD_MENTION,
                evidence=evidence,
            )
        )
    candidates.sort(key=lambda item: item.confidence, reverse=True)
    return candidates


def _find_node_similarity(
    page: DocPage,
    nodes: Mapping[str, NodeCatalogItem],
    *,
    top_k: int,
    threshold: float,
) -> List[NodeLinkCandidate]:
    page_tokens = _tokenize(page.text)
    ranked: List[Tuple[float, NodeCatalogItem, Set[str]]] = []
    for node in nodes.values():
        score, overlap = _token_overlap_score(page_tokens, node.token_set)
        if score < threshold:
            continue
        ranked.append((score, node, overlap))
    ranked.sort(key=lambda item: item[0], reverse=True)

    output: List[NodeLinkCandidate] = []
    for score, node, overlap in ranked[: max(0, top_k)]:
        confidence = min(0.55, 0.3 + score * 0.45)
        evidence = f"token_overlap={score:.2f}:{','.join(sorted(overlap)[:4])}"
        output.append(
            NodeLinkCandidate(
                doc_page_key=page.key,
                node_type=node.node_type,
                confidence=confidence,
                method=METHOD_SIMILARITY,
                evidence=evidence,
            )
        )
    return output


def _find_credential_similarity(
    page: DocPage,
    credentials: Mapping[str, CredentialCatalogItem],
    *,
    top_k: int,
    threshold: float,
) -> List[CredentialLinkCandidate]:
    page_tokens = _tokenize(page.text)
    ranked: List[Tuple[float, CredentialCatalogItem, Set[str]]] = []
    for credential in credentials.values():
        score, overlap = _token_overlap_score(page_tokens, credential.token_set)
        if score < threshold:
            continue
        ranked.append((score, credential, overlap))
    ranked.sort(key=lambda item: item[0], reverse=True)

    output: List[CredentialLinkCandidate] = []
    for score, credential, overlap in ranked[: max(0, top_k)]:
        confidence = min(0.52, 0.28 + score * 0.42)
        evidence = f"token_overlap={score:.2f}:{','.join(sorted(overlap)[:4])}"
        output.append(
            CredentialLinkCandidate(
                doc_page_key=page.key,
                credential_type=credential.credential_type,
                confidence=confidence,
                method=METHOD_SIMILARITY,
                evidence=evidence,
            )
        )
    return output


def _token_overlap_score(
    page_tokens: Set[str],
    candidate_tokens: Set[str],
) -> Tuple[float, Set[str]]:
    if not page_tokens or not candidate_tokens:
        return 0.0, set()
    overlap = page_tokens.intersection(candidate_tokens)
    if not overlap:
        return 0.0, set()
    score = len(overlap) / max(1, len(candidate_tokens))
    return score, overlap


def _match_doc_key(page_refs: Iterable[str], variants: Iterable[str]) -> str:
    normalized_variants = [variant.lower().strip() for variant in variants if variant]
    if not normalized_variants:
        return ""
    for ref in page_refs:
        ref_lower = ref.lower()
        for variant in normalized_variants:
            marker = f"/credentials/{variant}"
            if marker in ref_lower:
                return variant
            if ref_lower.endswith(f"/{variant}") and "credential" in ref_lower:
                return variant
    return ""


def _derived_confidence(node_confidence: float) -> float:
    return _clamp_confidence(max(0.65, min(0.92, node_confidence * 0.9)))


def _documentation_key_variants(value: str) -> Set[str]:
    raw = _as_text(value).lower().strip()
    if not raw:
        return set()
    variants: Set[str] = set()
    variants.add(raw.strip("/"))
    if "/" in raw:
        variants.add(raw.strip("/").split("/")[-1])
    normalized = normalize_reference(raw)
    if normalized:
        variants.add(normalized.strip("/"))
        host, path = _extract_host_path(normalized)
        if path:
            variants.add(path.strip("/").split("/")[-1])
    return {variant for variant in variants if variant}


def _display_name_mention(display_name: str, text_lower: str) -> str:
    candidate = _as_text(display_name).lower()
    if not candidate:
        return ""
    simple = _MULTISPACE_RE.sub(" ", candidate).strip()
    if not simple:
        return ""
    if simple in STOP_DISPLAY_NAMES:
        return ""
    if " " not in simple and len(simple) < 8:
        return ""
    pattern = r"\b" + re.escape(simple).replace(r"\ ", r"\s+") + r"\b"
    if re.search(pattern, text_lower):
        return simple
    return ""


def _catalog_tokens(identifier: str, display_name: str) -> Set[str]:
    tokens = set(_tokenize(identifier))
    tokens.update(_tokenize(display_name))
    return tokens


def _tokenize(text: str) -> Set[str]:
    lower = _as_text(text).lower()
    return {token for token in _TOKEN_RE.findall(lower) if len(token) >= 3}


def _normalize_path_component(path: str) -> str:
    unquoted = unquote(_as_text(path))
    cleaned = unquoted.replace("\\", "/").split("#", 1)[0].split("?", 1)[0]
    cleaned = cleaned.strip()
    if not cleaned:
        return "/"
    if not cleaned.startswith("/"):
        cleaned = "/" + cleaned
    cleaned = re.sub(r"/{2,}", "/", cleaned)
    if cleaned != "/":
        cleaned = cleaned.rstrip("/")
    return cleaned.lower()


def _looks_like_host_path(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z0-9.-]+\.[A-Za-z]{2,}(/.*)?$", value.strip()))


def _split_host_path(value: str) -> Tuple[str, str]:
    cleaned = value.strip()
    if "/" not in cleaned:
        return cleaned, "/"
    host, path = cleaned.split("/", 1)
    return host, "/" + path


def _looks_like_page_id(value: str) -> bool:
    cleaned = value.strip()
    if not cleaned:
        return False
    if "://" in cleaned or "/" in cleaned or "\\" in cleaned:
        return False
    return bool(re.match(r"^[a-zA-Z0-9._:-]+$", cleaned))


def _extract_host_path(normalized_reference: str) -> Tuple[str, str]:
    cleaned = normalized_reference.strip()
    if not cleaned:
        return "", ""
    if cleaned.startswith("/"):
        return "", cleaned
    first_slash = cleaned.find("/")
    if first_slash <= 0:
        return "", ""
    host = cleaned[:first_slash]
    if "." not in host:
        return "", ""
    return host, cleaned[first_slash:]


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value)


def _clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
