from app.indexing import extract_blocks, normalize_markdown, build_canonical_url


def test_normalize_markdown_strips_frontmatter_and_mdx():
    raw = (
        "---\n"
        "title: Demo\n"
        "---\n"
        "import Foo from './Foo';\n"
        "export const Bar = 1;\n"
        "# Heading\n\n"
        "Content here.\n"
    )
    cleaned = normalize_markdown(raw)
    assert "title: Demo" not in cleaned
    assert "import Foo" not in cleaned
    assert "export const" not in cleaned
    assert cleaned.startswith("# Heading")


def test_extract_blocks_tracks_sections():
    raw = (
        "# Title\n\n"
        "Intro paragraph.\n\n"
        "## Section A\n\n"
        "- item 1\n"
        "- item 2\n\n"
        "### Subsection\n\n"
        "Text under subsection.\n"
    )
    title, blocks = extract_blocks(raw, max_chars=500, min_chars=1, overlap=0)
    assert title == "Title"
    assert any(block.section == "Title / Section A" for block in blocks)
    assert any(block.section == "Title / Section A / Subsection" for block in blocks)


def test_build_canonical_url_trims_docs_root():
    url = build_canonical_url("docs/api/index.md", base_url="https://docs.n8n.io")
    assert url == "https://docs.n8n.io/api/"
