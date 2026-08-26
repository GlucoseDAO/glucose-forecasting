"""Typer CLI for building manuscript Markdown and PDF from LaTeX."""

from pathlib import Path
from typing import Annotated, Optional

import typer

from manuscript.convert import (
    MANUSCRIPT_TEX,
    TEMPLATE_TEX,
    latex_to_markdown,
    latex_to_pdf,
)

app = typer.Typer(
    name="manuscript",
    help="Build Markdown and PDF from manuscript LaTeX sources.",
    add_completion=False,
    pretty_exceptions_enable=False,
    no_args_is_help=True,
)


def _build(
    source: Path,
    markdown_output: Path | None,
    pdf_output: Path | None,
) -> None:
    try:
        markdown_path = latex_to_markdown(source, markdown_output)
        pdf_path = latex_to_pdf(source, pdf_output)
    except (FileNotFoundError, RuntimeError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Wrote {markdown_path}")
    typer.echo(f"Wrote {pdf_path}")


@app.command("template")
def build_template(
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output Markdown file. Defaults to docs/manuscript/template.md.",
        ),
    ] = None,
    pdf_output: Annotated[
        Optional[Path],
        typer.Option(
            "--pdf-output",
            help="Output PDF file. Defaults to docs/manuscript/template.pdf.",
        ),
    ] = None,
) -> None:
    """Build Markdown and PDF from the EASRP template."""
    _build(TEMPLATE_TEX, output, pdf_output)


@app.command("manuscript")
def build_manuscript(
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output Markdown file. Defaults to docs/manuscript/manuscript.md.",
        ),
    ] = None,
    pdf_output: Annotated[
        Optional[Path],
        typer.Option(
            "--pdf-output",
            help="Output PDF file. Defaults to docs/manuscript/manuscript.pdf.",
        ),
    ] = None,
) -> None:
    """Build Markdown and PDF from the paper LaTeX."""
    _build(MANUSCRIPT_TEX, output, pdf_output)


if __name__ == "__main__":
    app()
