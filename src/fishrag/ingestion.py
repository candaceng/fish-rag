"""Reviewed local HTML ingestion shared by FishRAG's experiment notebooks."""

import hashlib

from bs4 import BeautifulSoup
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.utils import get_tokenizer


def content_kind(source, heading_path):
    if source["source_type"] == "regulation_summary":
        return "regulation_summary"
    if "Rules and regulations" in heading_path:
        return "regulatory_guidance"
    if "Smallmouth and largemouth bass" in heading_path:
        return "mixed_biology_and_technique"
    if heading_path and heading_path[0] in {
        "Fishing rods, reels and line",
        "Tackle and other gear",
        "Knots and casting",
        "Tips and techniques",
    }:
        return "technique_advice"
    return "general_guide_context"


def block_text(tag):
    # Keep list items and table rows visibly separate without duplicating text.
    block = BeautifulSoup(str(tag), "html.parser")
    for element in block.find_all(["li", "p", "tr"]):
        element.insert_before("\n")
        element.insert_after("\n")
    return "\n".join(
        " ".join(line.split())
        for line in block.get_text(" ").splitlines()
        if line.strip()
    )


def extract_sections(body):
    # Work on a copy: the original downloaded HTML is never changed.
    body = BeautifulSoup(str(body), "html.parser")
    # Definition terms are real source labels, represented as subsection headings.
    for term in body.select("dl > dt"):
        term.name = "h3"
    for definition_list in body.find_all("dl"):
        definition_list.unwrap()
    # Only take outer blocks so nested lists and table cells are not duplicated.
    blocks = {"p", "ul", "ol", "dd", "table"}
    headings = {}
    lines = []
    section_anchor = None
    sections = []
    for tag in body.find_all(["h2", "h3", "h4", "h5", "h6", *sorted(blocks)]):
        if any(parent.name in blocks for parent in tag.parents if parent is not body):
            continue
        if tag.name.startswith("h"):
            if lines:
                sections.append(
                    (list(headings.values()), section_anchor, "\n\n".join(lines))
                )
            level = int(tag.name[1])
            headings = {key: value for key, value in headings.items() if key < level}
            headings[level] = tag.get_text(" ", strip=True)
            section_anchor = tag.get("id")  # Do not invent missing anchors.
            lines = []
        else:
            text = block_text(tag)
            if text:
                lines.append(text)
    if lines:
        sections.append((list(headings.values()), section_anchor, "\n\n".join(lines)))
    return sections


def load_documents(root, manifest):
    """Load the reviewed snapshots with exact-byte and title validation."""
    sources = manifest["sources"]
    soups = {}
    for source in sources:
        path = root / source["local_path"]
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != source["sha256"]:
            raise ValueError(f"Snapshot changed: {path.name}; review the manifest.")
        soup = BeautifulSoup(raw.decode("utf-8"), "html.parser")
        main = soup.find("main")
        heading = main.find("h1") if main else None
        if not heading or heading.get_text(" ", strip=True) != source["title"]:
            raise ValueError(f"Source title differs: {path.name}")
        if main.select_one(".body-field") is None:
            raise ValueError(f"Source body missing: {path.name}")
        soups[source["source_id"]] = soup
    documents = []
    for source in sources:
        body = soups[source["source_id"]].find("main").select_one(".body-field")
        sections = extract_sections(body)
        section_text = {tuple(path): text for path, _, text in sections}
        for section_index, (heading_path, anchor, text) in enumerate(sections):
            section = " > ".join(heading_path) or "Introduction (before first heading)"
            metadata = {
                key: source[key]
                for key in (
                    "source_id",
                    "title",
                    "url",
                    "publisher",
                    "geographic_scope",
                    "source_type",
                    "publication_date",
                    "publication_year",
                    "updated_date",
                    "regulation_year",
                    "access_date",
                    "page",
                    "sha256",
                    "attribution",
                    "terms_url",
                )
            }
            # Carry explicit parent qualifiers into regulation subsections.
            # Sibling exceptions still require consulting their own sections.
            context = []
            if source["source_type"] == "regulation_summary":
                for depth in range(1, len(heading_path)):
                    parent_path = tuple(heading_path[:depth])
                    if parent_path in section_text:
                        context.append(
                            (" > ".join(parent_path), section_text[parent_path])
                        )
            context_text = "\n\n".join(
                f"Parent section: {heading}\n{body_text}"
                for heading, body_text in context
            )
            metadata.update(
                parent_context_sections=" | ".join(heading for heading, _ in context),
                section=section,
                section_url=f"{source['url']}#{anchor}" if anchor else source["url"],
                content_kind=content_kind(source, heading_path),
            )
            documents.append(
                Document(
                    id_=f"{source['source_id']}:{section_index}",
                    text="\n\n".join(
                        part for part in (section, context_text, text) if part
                    ),
                    metadata=metadata,
                    # Retain provenance without spending the text budget on metadata.
                    excluded_embed_metadata_keys=list(metadata),
                    excluded_llm_metadata_keys=list(metadata),
                )
            )

    return documents


def chunk_documents(documents, chunk_size=512, chunk_overlap=64):
    """Apply the reviewed local SentenceSplitter without model calls."""
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        tokenizer=get_tokenizer(),
    )
    return splitter.get_nodes_from_documents(documents)
