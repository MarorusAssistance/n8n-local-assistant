from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import re


_FRONTMATTER_RE = re.compile(r"\A---[\s\S]*?---\s*")
_MDX_IMPORT_RE = re.compile(r"^\s*import\s.+?;\s*$", re.MULTILINE)
_MDX_EXPORT_RE = re.compile(r"^\s*export\s.+?;\s*$", re.MULTILINE)
_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_TABLE_SEP_RE = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*$")
_LIST_RE = re.compile(r"^\s*(?:[-*+]|(\d+)[.)])\s+")
_QUOTE_RE = re.compile(r"^\s*>\s?")


@dataclass
class MarkdownBlock:
    text: str
    section: Optional[str]
    title: Optional[str]
    kind: str


def fnv1a32(text: str) -> str:
    h = 0x811C9DC5
    for ch in text:
        h ^= ord(ch)
        h = (h + ((h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24))) & 0xFFFFFFFF
    return f"{h:08x}"


def normalize_markdown(text: str) -> str:
    cleaned = str(text or "")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _FRONTMATTER_RE.sub("", cleaned)
    cleaned = _MDX_IMPORT_RE.sub("", cleaned)
    cleaned = _MDX_EXPORT_RE.sub("", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    return cleaned.strip()


def build_canonical_url(rel_path: str, base_url: str, docs_root: str = "docs") -> str:
    url_path = (rel_path or "").replace("\\", "/").lstrip("/")
    root = (docs_root or "").strip("/").replace("\\", "/")
    if root:
        prefix = f"{root}/"
        if url_path.lower().startswith(prefix.lower()):
            url_path = url_path[len(prefix):]
    elif url_path.lower().startswith("docs/"):
        url_path = url_path[5:]
    url_path = re.sub(r"\.(md|mdx)$", "", url_path, flags=re.IGNORECASE)
    url_path = re.sub(r"/index$", "/", url_path, flags=re.IGNORECASE)
    base = base_url.rstrip("/")
    if url_path:
        return f"{base}/{url_path.lstrip('/')}"
    return base + "/"


def extract_blocks(
    text: str,
    default_title: Optional[str] = None,
    max_chars: int = 1600,
    min_chars: int = 200,
    overlap: int = 120,
) -> Tuple[Optional[str], List[MarkdownBlock]]:
    cleaned = normalize_markdown(text)
    raw_blocks, doc_title = _tokenize_markdown(cleaned)
    if not doc_title and default_title:
        doc_title = default_title
    for block in raw_blocks:
        block.title = doc_title

    merged = _merge_blocks(raw_blocks, max_chars=max_chars, min_chars=min_chars)
    final = _split_large_blocks(merged, max_chars=max_chars, overlap=overlap)
    return doc_title, final


def _tokenize_markdown(text: str) -> Tuple[List[MarkdownBlock], Optional[str]]:
    lines = text.splitlines()
    blocks: List[MarkdownBlock] = []
    section_stack: List[str] = []
    doc_title: Optional[str] = None

    i = 0
    while i < len(lines):
        line = lines[i]
        heading_match = _HEADING_RE.match(line)
        if heading_match:
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            heading_text = re.sub(r"\s+#*$", "", heading_text).strip()
            if level == 1 and not doc_title and heading_text:
                doc_title = heading_text
            if heading_text:
                if len(section_stack) >= level:
                    section_stack = section_stack[: level - 1]
                while len(section_stack) < level - 1:
                    section_stack.append("")
                section_stack.append(heading_text)
            i += 1
            continue

        fence_match = _FENCE_RE.match(line)
        if fence_match:
            fence = fence_match.group(1)
            block_lines = [line]
            i += 1
            while i < len(lines):
                block_lines.append(lines[i])
                if lines[i].strip().startswith(fence):
                    i += 1
                    break
                i += 1
            blocks.append(
                MarkdownBlock(
                    text="\n".join(block_lines).strip(),
                    section=_section_label(section_stack),
                    title=doc_title,
                    kind="code",
                )
            )
            continue

        if _is_table_start(lines, i):
            block_lines = [line]
            i += 1
            while i < len(lines):
                next_line = lines[i]
                if not next_line.strip():
                    break
                if "|" not in next_line:
                    break
                block_lines.append(next_line)
                i += 1
            blocks.append(
                MarkdownBlock(
                    text="\n".join(block_lines).strip(),
                    section=_section_label(section_stack),
                    title=doc_title,
                    kind="table",
                )
            )
            continue

        if _LIST_RE.match(line):
            block_lines = [line]
            i += 1
            while i < len(lines):
                next_line = lines[i]
                if _is_hard_boundary(lines, i):
                    break
                if not next_line.strip():
                    lookahead = lines[i + 1] if i + 1 < len(lines) else ""
                    if _LIST_RE.match(lookahead) or lookahead.startswith(" "):
                        block_lines.append(next_line)
                        i += 1
                        continue
                    break
                if _LIST_RE.match(next_line) or next_line.startswith(" "):
                    block_lines.append(next_line)
                    i += 1
                    continue
                break
            blocks.append(
                MarkdownBlock(
                    text="\n".join(block_lines).strip(),
                    section=_section_label(section_stack),
                    title=doc_title,
                    kind="list",
                )
            )
            continue

        if _QUOTE_RE.match(line):
            block_lines = [line]
            i += 1
            while i < len(lines):
                next_line = lines[i]
                if _is_hard_boundary(lines, i):
                    break
                if _QUOTE_RE.match(next_line) or not next_line.strip():
                    block_lines.append(next_line)
                    i += 1
                    continue
                break
            blocks.append(
                MarkdownBlock(
                    text="\n".join(block_lines).strip(),
                    section=_section_label(section_stack),
                    title=doc_title,
                    kind="quote",
                )
            )
            continue

        if not line.strip():
            i += 1
            continue

        block_lines = [line]
        i += 1
        while i < len(lines):
            next_line = lines[i]
            if not next_line.strip():
                break
            if _is_block_boundary(lines, i):
                break
            block_lines.append(next_line)
            i += 1
        blocks.append(
            MarkdownBlock(
                text="\n".join(block_lines).strip(),
                section=_section_label(section_stack),
                title=doc_title,
                kind="paragraph",
            )
        )

    return blocks, doc_title


def _section_label(stack: Iterable[str]) -> Optional[str]:
    parts = [part for part in stack if part]
    if not parts:
        return None
    return " / ".join(parts)


def _is_block_boundary(lines: List[str], index: int) -> bool:
    line = lines[index]
    if _HEADING_RE.match(line):
        return True
    if _FENCE_RE.match(line):
        return True
    if _is_table_start(lines, index):
        return True
    if _LIST_RE.match(line):
        return True
    if _QUOTE_RE.match(line):
        return True
    return False


def _is_hard_boundary(lines: List[str], index: int) -> bool:
    line = lines[index]
    if _HEADING_RE.match(line):
        return True
    if _FENCE_RE.match(line):
        return True
    if _is_table_start(lines, index):
        return True
    return False


def _is_table_start(lines: List[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    line = lines[index]
    if "|" not in line:
        return False
    return bool(_TABLE_SEP_RE.match(lines[index + 1]))


def _merge_blocks(
    blocks: List[MarkdownBlock],
    max_chars: int,
    min_chars: int,
) -> List[MarkdownBlock]:
    merged: List[MarkdownBlock] = []
    buffer: Optional[MarkdownBlock] = None

    for block in blocks:
        if not block.text.strip():
            continue
        if buffer is None:
            buffer = block
            continue

        same_section = buffer.section == block.section
        combined_len = len(buffer.text) + 2 + len(block.text)
        should_merge = (
            same_section
            and combined_len <= max_chars
            and (len(buffer.text) < min_chars or len(block.text) < min_chars)
        )
        if should_merge:
            buffer.text = buffer.text.rstrip() + "\n\n" + block.text.lstrip()
            continue

        merged.append(buffer)
        buffer = block

    if buffer is not None:
        merged.append(buffer)

    if len(merged) >= 2:
        last = merged[-1]
        prev = merged[-2]
        combined_len = len(prev.text) + 2 + len(last.text)
        if (
            prev.section == last.section
            and len(last.text) < min_chars
            and combined_len <= max_chars
        ):
            prev.text = prev.text.rstrip() + "\n\n" + last.text.lstrip()
            merged.pop()

    return merged


def _split_large_blocks(
    blocks: List[MarkdownBlock],
    max_chars: int,
    overlap: int,
) -> List[MarkdownBlock]:
    if max_chars <= 0:
        return blocks
    final: List[MarkdownBlock] = []
    for block in blocks:
        if len(block.text) <= max_chars:
            final.append(block)
            continue
        parts = _split_text_natural(block.text, max_chars=max_chars, overlap=overlap)
        for part in parts:
            final.append(
                MarkdownBlock(
                    text=part,
                    section=block.section,
                    title=block.title,
                    kind=block.kind,
                )
            )
    return final


def _split_text_natural(text: str, max_chars: int, overlap: int) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    chunks: List[str] = []
    start = 0
    text_len = len(text)
    safe_overlap = max(0, min(overlap, max_chars - 1))

    while start < text_len:
        end = min(start + max_chars, text_len)
        if end < text_len:
            split_at = text.rfind("\n\n", start, end)
            if split_at < start + int(max_chars * 0.5):
                split_at = text.rfind("\n", start, end)
            if split_at < start + int(max_chars * 0.5):
                split_at = end
            chunk = text[start:split_at].strip()
            if not chunk:
                chunk = text[start:end].strip()
                split_at = end
        else:
            split_at = end
            chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if split_at >= text_len:
            break

        next_start = split_at - safe_overlap
        if next_start <= start:
            next_start = split_at
        start = next_start

    return chunks
