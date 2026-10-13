"""
Build a single assignment PDF from design docs, traces,
test output, and key source files.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "deliverable" / "assignment_deliverable.pdf"

SECTIONS = [
    (
        "1. State and memory design decision and rationale",
        PROJECT_ROOT / "docs" / "state_memory_design.md",
    ),
    (
        "2. Tracing implementation: trace record schema",
        PROJECT_ROOT / "src" / "underwriting" / "observability" / "models.py",
    ),
    (
        "3. Tracing implementation: PII redaction",
        PROJECT_ROOT / "src" / "underwriting" / "observability" / "payloads.py",
    ),
    (
        "4. Full pipeline / state-sharing mechanism and trace wiring",
        PROJECT_ROOT / "src" / "underwriting" / "workflow" / "orchestrator.py",
    ),
    (
        "5. Cost and latency analysis",
        PROJECT_ROOT / "docs" / "cost_latency_analysis.md",
    ),
    (
        "6. Test and coverage output",
        PROJECT_ROOT / "artifacts" / "lab_evidence" / "unit_tests.txt",
    ),
    (
        "7. Trace: case_01",
        PROJECT_ROOT / "artifacts" / "lab_evidence" / "case_01_trace.txt",
    ),
    (
        "8. Trace: case_02",
        PROJECT_ROOT / "artifacts" / "lab_evidence" / "case_02_trace.txt",
    ),
    (
        "9. Trace: case_03",
        PROJECT_ROOT / "artifacts" / "lab_evidence" / "case_03_trace.txt",
    ),
    (
        "10. Simulated failure (retry then escalate)",
        PROJECT_ROOT / "artifacts" / "lab_evidence" / "failure_trace.txt",
    ),
    (
        "11. CI workflow",
        PROJECT_ROOT / ".github" / "workflows" / "ci.yml",
    ),
]


def pdf_escape(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
    )


def wrap_file(path: Path, limit: int = 22000) -> list[str]:
    if not path.exists():
        return [f"[missing file: {path}]"]

    text = path.read_text(encoding="utf-8")
    if len(text) > limit:
        text = text[:limit] + "\n... [truncated for PDF size] ..."

    wrapped: list[str] = []
    for raw_line in text.splitlines() or [""]:
        wrapped.extend(
            textwrap.wrap(
                raw_line,
                width=92,
                replace_whitespace=False,
                drop_whitespace=False,
            )
            or [""]
        )
    return wrapped


def build_pages(lines: list[str], lines_per_page: int = 52) -> list[list[str]]:
    pages: list[list[str]] = []
    for index in range(0, len(lines), lines_per_page):
        pages.append(lines[index:index + lines_per_page])
    return pages or [[""]]


def write_pdf(pages: list[list[str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    catalog_id = 1
    pages_id = 2
    font_id = 3
    first_page_id = 4

    page_ids = list(range(first_page_id, first_page_id + len(pages)))
    content_ids = list(
        range(first_page_id + len(pages), first_page_id + 2 * len(pages))
    )

    parts: list[bytes] = [b"%PDF-1.4\n"]
    xref_positions = [0]

    def emit(obj_id: int, payload: bytes) -> None:
        xref_positions.append(sum(len(part) for part in parts))
        parts.append(f"{obj_id} 0 obj\n".encode("latin-1") + payload + b"\nendobj\n")

    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)

    emit(
        catalog_id,
        f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode("latin-1"),
    )
    emit(
        pages_id,
        (
            f"<< /Type /Pages /Count {len(pages)} "
            f"/Kids [{kids}] >>"
        ).encode("latin-1"),
    )
    emit(
        font_id,
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>",
    )

    for page_id, content_id in zip(page_ids, content_ids, strict=True):
        emit(
            page_id,
            (
                f"<< /Type /Page /Parent {pages_id} 0 R "
                f"/MediaBox [0 0 612 792] "
                f"/Contents {content_id} 0 R "
                f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>"
            ).encode("latin-1"),
        )

    for content_id, page_lines in zip(content_ids, pages, strict=True):
        stream_lines = ["BT", "/F1 8 Tf", "36 760 Td", "10 TL"]
        for line in page_lines:
            stream_lines.append(f"({pdf_escape(line)}) '")
        stream_lines.append("ET")
        stream = "\n".join(stream_lines).encode("latin-1", errors="replace")
        emit(
            content_id,
            (
                f"<< /Length {len(stream)} >>\nstream\n".encode("latin-1")
                + stream
                + b"\nendstream"
            ),
        )

    xref_start = sum(len(part) for part in parts)
    object_count = 3 + 2 * len(pages)
    xref = [b"xref\n", f"0 {object_count + 1}\n".encode("latin-1")]
    xref.append(b"0000000000 65535 f \n")
    # offsets[1] is first object; emit stored starting at index 1
    # We stored positions as we emitted, skipping the header at parts[0]
    recorded = []
    running = 0
    # Recompute from parts: first part is header
    running = len(parts[0])
    for part in parts[1:]:
        recorded.append(running)
        running += len(part)

    for position in recorded:
        xref.append(f"{position:010d} 00000 n \n".encode("latin-1"))

    trailer = (
        f"trailer\n<< /Size {object_count + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_start}\n%%EOF\n"
    ).encode("latin-1")

    output.write_bytes(b"".join(parts) + b"".join(xref) + trailer)


def main() -> None:
    lines: list[str] = [
        "Multi-agent Underwriting Assignment Packet",
        "Design, tracing, PII handling, traces, cost, failure, CI",
        "",
        "Standalone unit tests live in tests/unit/test_*_agent.py.",
        "Paste terminal screenshots of pytest and GitHub Actions",
        "around the captured text below if the rubric requires photos.",
        "",
    ]

    for title, path in SECTIONS:
        lines.append("=" * 72)
        lines.append(title)
        lines.append(f"Source: {path.relative_to(PROJECT_ROOT)}")
        lines.append("=" * 72)
        lines.extend(wrap_file(path))
        lines.append("")

    write_pdf(build_pages(lines), OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
