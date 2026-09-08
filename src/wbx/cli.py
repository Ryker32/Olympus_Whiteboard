"""``wbx`` command line."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .io import HeicSupportMissing
from .people import MediaPipeMissing
from .pipeline import Params, process_photo
from .rectify import parse_corners
from .sheet import build_sheet

app = typer.Typer(add_completion=False, help="Whiteboard quote extractor.")


@app.command()
def extract(
    photos: list[Path] = typer.Argument(..., exists=True, dir_okay=False, help="Board photos."),
    out: Path = typer.Option(..., "-o", "--out", help="Output directory."),
    debug: bool = typer.Option(False, "--debug", help="Write numbered intermediates to OUT/<photo>/debug/."),
    no_rectify: bool = typer.Option(False, "--no-rectify", help="Skip board detection and perspective warp."),
    corners: Optional[str] = typer.Option(None, "--corners", metavar="x1,y1,x2,y2,x3,y3,x4,y4",
                                          help="Manual board corners, overriding detection."),
    alpha_floor: float = typer.Option(0.06, help="Coverage below this is treated as bare board."),
    sat_weight: float = typer.Option(0.7, help="Weight of the saturation cue; catches yellow and pastels."),
    min_area: int = typer.Option(150, help="Drop connected components smaller than this (px)."),
    min_quote_area: int = typer.Option(800, help="Minimum stroke area for a merged blob to count as a quote."),
    merge_radius: Optional[int] = typer.Option(
        None, help="Dilation radius joining strokes into one quote (px). Default: 5x the stroke width."),
    boost_sat: float = typer.Option(0.0, help="Punch up recovered saturation. Off by default, on purpose."),
    ghost_max_alpha: float = typer.Option(0.35, help="Eraser-ghost test: peak alpha below this..."),
    ghost_max_sat: float = typer.Option(0.10, help="...and mean saturation below this, means ghost."),
    bg_method: str = typer.Option("median", help="Background estimator: 'median' or 'close' (fat-marker fallback)."),
    remove_people: bool = typer.Option(False, "--remove-people", help="Mask out people (needs the 'people' extra)."),
    bleed: bool = typer.Option(True, help="Extend ink colour under transparent pixels to stop scaling halos."),
) -> None:
    """Extract transparent quote stickers from board photos."""
    params = Params(
        alpha_floor=alpha_floor,
        sat_weight=sat_weight,
        min_area=min_area,
        min_quote_area=min_quote_area,
        merge_radius=merge_radius,
        boost_sat=boost_sat,
        ghost_max_alpha=ghost_max_alpha,
        ghost_max_sat=ghost_max_sat,
        bg_method=bg_method,
        rectify=not no_rectify,
        corners=parse_corners(corners).ravel().tolist() if corners else None,
        remove_people=remove_people,
        bleed=bleed,
    )

    total = 0
    for photo in photos:
        try:
            result = process_photo(photo, out, params, debug=debug)
        except (HeicSupportMissing, MediaPipeMissing) as exc:
            # missing optional extras are a setup problem, not a crash
            typer.secho(str(exc), fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from None
        for w in result.warnings:
            typer.secho(f"  ! {photo.name}: {w}", fg=typer.colors.YELLOW, err=True)
        total += result.n_quotes
        typer.echo(f"{photo.name}: {result.n_quotes} quote(s) -> {result.out_dir}")

    typer.secho(f"{total} sticker(s) in {out}", fg=typer.colors.GREEN)
    typer.echo(f"Review them with: wbx sheet {out}")


@app.command()
def sheet(
    out: Path = typer.Argument(..., exists=True, file_okay=False, help="An extract output directory."),
) -> None:
    """Render a QA contact sheet of every sticker on light and dark checkerboards."""
    dest = build_sheet(out)
    typer.secho(f"wrote {dest}", fg=typer.colors.GREEN)


if __name__ == "__main__":  # pragma: no cover
    app()
