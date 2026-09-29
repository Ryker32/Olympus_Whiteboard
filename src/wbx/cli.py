"""``wbx`` command line."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .album import build_album
from .io import HeicSupportMissing
from .people import MediaPipeMissing
from .pipeline import Keep, Params, process_photo
from .recut import recut as recut_crops
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
        None, help="Slack around each stroke joining it into a quote (px). Default: 0.7x letter height."),
    keep: Optional[Keep] = typer.Option(
        None, "--keep", case_sensitive=False,
        help="Which quotes to export: 'all' (a board shot), 'main' (a close-up: the centred "
             "quote, dropping neighbours clipped at the frame edge), or 'merged' (one sticker "
             "from every stroke). Default: merged for input with a crop mask, else all."),
    boost_sat: float = typer.Option(0.0, help="Punch up recovered saturation. Off by default, on purpose."),
    ghost_max_alpha: float = typer.Option(0.35, help="Eraser-ghost test: peak alpha below this..."),
    ghost_max_sat: float = typer.Option(0.10, help="...and mean saturation below this, means ghost."),
    bg_method: str = typer.Option("median", help="Background estimator: 'median' or 'close' (fat-marker fallback)."),
    bg_refine: bool = typer.Option(True, "--bg-refine/--no-bg-refine",
                                   help="Re-estimate the board lighting with the ink masked out."),
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
        keep=keep,
        boost_sat=boost_sat,
        ghost_max_alpha=ghost_max_alpha,
        ghost_max_sat=ghost_max_sat,
        bg_method=bg_method,
        bg_refine=bg_refine,
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
def recut(
    crops: list[Path] = typer.Argument(..., exists=True, dir_okay=False, help="Hand-made crops (RGBA)."),
    photos: Path = typer.Option(..., "-p", "--photos", exists=True, file_okay=False,
                                help="Directory of the original full-resolution photos."),
    out: Path = typer.Option(..., "-o", "--out", help="Where to write the re-cut regions."),
    pattern: str = typer.Option("*.jpg", help="Which files in --photos to search."),
) -> None:
    """Re-cut hand-made crops from the originals, keeping the segmentation."""
    originals = sorted(Path(photos).glob(pattern))
    if not originals:
        typer.secho(f"no {pattern} under {photos}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    typer.echo(f"matching {len(crops)} crop(s) against {len(originals)} photo(s)...")
    results = recut_crops(list(crops), originals, out)
    found = [r for r in results if r.written]
    for r in results:
        if not r.written:
            typer.secho(f"  ! {r.crop.name}: no confident match (best {r.inliers} inliers)",
                        fg=typer.colors.YELLOW, err=True)
    typer.secho(f"{len(found)}/{len(results)} re-cut -> {out}", fg=typer.colors.GREEN)
    if found:
        gains = sorted(r.gain for r in found)
        typer.echo(f"resolution gain: median {gains[len(gains) // 2]:.2f}x")
        typer.echo(f"Now run: wbx extract {out}/*.png -o OUT")


@app.command()
def album(
    out: Path = typer.Argument(..., exists=True, file_okay=False, help="An extract output directory."),
    dest: Path = typer.Option(..., "-o", "--out", help="Folder to write the flat copies into."),
    prefix: str = typer.Option("sticker", help="Filename stem; files are PREFIX_01.png and so on."),
) -> None:
    """Copy every sticker into one flat folder, with an index, ready to hand over."""
    rows = build_album(out, dest, prefix=prefix)
    dupes = [r for r in rows if r.looks_like]
    typer.secho(f"{len(rows)} sticker(s) -> {dest}", fg=typer.colors.GREEN)
    for r in dupes:
        typer.secho(f"  ? {r.file} looks like the same quote as {r.looks_like}",
                    fg=typer.colors.YELLOW)
    if dupes:
        typer.echo(f"{len(dupes)} possible duplicate(s) flagged in index.csv; nothing was dropped.")


@app.command()
def sheet(
    out: Path = typer.Argument(..., exists=True, file_okay=False, help="An extract output directory."),
) -> None:
    """Render a QA contact sheet of every sticker on light and dark checkerboards."""
    dest = build_sheet(out)
    typer.secho(f"wrote {dest}", fg=typer.colors.GREEN)


if __name__ == "__main__":  # pragma: no cover
    app()
