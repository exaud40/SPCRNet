# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 20:23:56 2026

@author: Exaud
"""

from __future__ import annotations
import argparse
import csv
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import numpy as np

MDMT_SPLITS = ("train", "val", "test")
VIEWS = ("1", "2")
INDEX_FIELDS = ["split", "seq", "view", "frame", "image", "n_view", "n_view_clear", "n_union", "n_overlap",
                "n_union_person", "n_union_car", "n_union_other"]


def parse_tracks(xml_path: str | Path, centers: bool = False, boxes: bool = False) -> dict[int, dict[int, tuple]]:
    """{xml frame: {track id: (label, occluded)}} for boxes with outside == 0 (target visible in this view).
    With centers=True the value is (label, occluded, cx, cy); with boxes=True (label, occluded, xtl, ytl, xbr, ybr),
    in pixels of the original frame."""
    frames: dict[int, dict[int, tuple]] = defaultdict(dict)
    tid, label = None, None
    for event, el in ET.iterparse(str(xml_path), events=("start", "end")):
        if el.tag == "track" and event == "start":
            tid, label = int(el.get("id")), el.get("label", "?")
        elif el.tag == "box" and event == "end":
            if int(el.get("outside", 0)) == 0:
                v = (label, int(el.get("occluded", 0)))
                if boxes:
                    v += (float(el.get("xtl")), float(el.get("ytl")), float(el.get("xbr")), float(el.get("ybr")))
                elif centers:
                    v += (0.5 * (float(el.get("xtl")) + float(el.get("xbr"))),
                          0.5 * (float(el.get("ytl")) + float(el.get("ybr"))))
                frames[int(el.get("frame"))][tid] = v
            el.clear()
        elif el.tag == "track" and event == "end":
            el.clear()
    return dict(frames)


def image_numbers(folder: Path) -> list[int]:
    return sorted(int(p.stem) for p in folder.glob("*.jpg") if p.stem.isdigit())


def best_offset(img_nums: list[int], xml_frames: set[int]) -> tuple[int, float]:
    """Image number n shows xml frame n - offset. Candidates: 1 (absolute numbering, frame 0 = 00000001.jpg)
    and the first image number (xml counts from the first image on disk). Returns (offset, coverage)."""
    best = (1, -1.0)
    for off in sorted({1, img_nums[0]}):
        cov = float(np.mean([(n - off) in xml_frames for n in img_nums]))
        if cov > best[1]:
            best = (off, cov)
    return best


@dataclass
class SeqPair:
    split: str
    seq: str
    dirs: dict[str, Path]         # view -> image folder
    xmls: dict[str, Path]         # view -> xml


def find_pairs(root: str | Path) -> list[SeqPair]:
    root = Path(root)
    pairs = []
    for split in MDMT_SPLITS:
        seqs = defaultdict(dict)
        for v in VIEWS:
            d = root / split / v
            if not d.is_dir():
                continue
            for s in d.iterdir():
                m = re.fullmatch(r"(\d+)-(\d+)", s.name)
                if s.is_dir() and m and m.group(2) == v:
                    seqs[m.group(1)][v] = s
        for seq, dirs in sorted(seqs.items(), key=lambda kv: int(kv[0])):
            xmls = {v: root / "new_xml" / v / f"{seq}-{v}.xml" for v in VIEWS}
            if set(dirs) == set(VIEWS) and all(x.exists() for x in xmls.values()):
                pairs.append(SeqPair(split, seq, dirs, xmls))
            else:
                print(f"[mdmt] skipping {split}/{seq}: views {sorted(dirs)}, xml present "
                      f"{[x.exists() for x in xmls.values()]}")
    return pairs


def _cls(label: str) -> str:
    label = label.lower()
    return "person" if label in ("person", "pedestrian", "people") else "car" if label == "car" else "other"


def index_pair(pair: SeqPair, root: Path) -> tuple[list[dict], dict]:
    """Rows for every frame present in BOTH views, plus an alignment record."""
    tracks = {v: parse_tracks(pair.xmls[v]) for v in VIEWS}
    nums = {v: image_numbers(pair.dirs[v]) for v in VIEWS}
    common = sorted(set(nums["1"]) & set(nums["2"]))
    if not common:
        return [], {"seq": pair.seq, "frames": 0}
    align = {}
    for v in VIEWS:
        off, cov = best_offset(common, set(tracks[v]))
        align[v] = (off, cov)
    rows = []
    for n in common:
        vis = {v: tracks[v].get(n - align[v][0], {}) for v in VIEWS}
        union = {**vis["1"], **vis["2"]}
        both = set(vis["1"]) & set(vis["2"])
        by_cls = Counter(_cls(lab) for lab, _ in union.values())
        for v in VIEWS:
            rows.append({
                "split": pair.split, "seq": pair.seq, "view": int(v) - 1, "frame": n,
                "image": (pair.dirs[v] / f"{n:08d}.jpg").relative_to(root).as_posix(),
                "n_view": len(vis[v]), "n_view_clear": sum(1 for _, occ in vis[v].values() if occ == 0),
                "n_union": len(union), "n_overlap": len(both),
                "n_union_person": by_cls["person"], "n_union_car": by_cls["car"], "n_union_other": by_cls["other"],
            })
    rec = {"seq": pair.seq, "split": pair.split, "frames": len(common),
           "offset_v1": align["1"][0], "coverage_v1": align["1"][1],
           "offset_v2": align["2"][0], "coverage_v2": align["2"][1]}
    return rows, rec


def build_index(root: str | Path) -> tuple[list[dict], list[dict]]:
    root = Path(root)
    rows, recs = [], []
    for p in find_pairs(root):
        r, rec = index_pair(p, root)
        rows += r
        recs.append(rec)
        print(f"[mdmt] {p.split:5s} seq {p.seq:>3s}: {rec['frames']:4d} paired frames, "
              f"xml coverage {rec.get('coverage_v1', 0):.0%}/{rec.get('coverage_v2', 0):.0%}", flush=True)
    return rows, recs


def write_index(rows: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=INDEX_FIELDS)
        w.writeheader()
        w.writerows(rows)


def read_index(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return [{k: (v if k in ("split", "seq", "image") else int(v)) for k, v in r.items()} for r in csv.DictReader(fh)]


def quantile_edges(counts: np.ndarray, n_bins: int) -> np.ndarray:
    """Interior bin edges (integers, strictly increasing) giving roughly balanced classes:
    class c = number of edges <= count."""
    qs = np.quantile(counts, np.linspace(0, 1, n_bins + 1)[1:-1])
    edges = []
    for q in np.ceil(qs).astype(int):
        if not edges or q > edges[-1]:
            edges.append(int(q))
    return np.array(edges, dtype=int)


def count_bin(counts, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges, np.asarray(counts), side="right")


def label_report(rows: list[dict], n_bins_list=(4, 6, 8)) -> str:
    """Statistics that decide the label space. 'Ceiling' = accuracy of a PERFECT single-view counter,
    i.e. how often the view's own count already lands in the right bin."""
    a = {k: np.array([r[k] for r in rows]) for k in ("n_view", "n_view_clear", "n_union", "n_overlap")}
    split = np.array([r["split"] for r in rows])
    lines = [f"rows (view x frame): {len(rows)}   per split: {dict(Counter(split.tolist()))}",
             "",
             "counts per frame        mean   p10   p50   p90   max",
             *[f"  {k:18s} {a[k].mean():6.1f} {np.percentile(a[k], 10):5.0f} {np.percentile(a[k], 50):5.0f} "
               f"{np.percentile(a[k], 90):5.0f} {a[k].max():5d}" for k in a],
             f"  share of scene targets seen by this view: {np.mean(a['n_view'] / np.maximum(a['n_union'], 1)):.2f}",
             f"  share of this view's targets that are occluded: "
             f"{1 - a['n_view_clear'].sum() / max(a['n_view'].sum(), 1):.2f}",
             "",
             "Task A: label = bin(scene count = union of both views). Edges from TRAIN quantiles.",
             "   C   edges                          class shares (train)            ceiling: own count | clear-only"]
    train = split == "train"
    for C in n_bins_list:
        e = quantile_edges(a["n_union"][train], C)
        y = count_bin(a["n_union"], e)
        share = np.bincount(y[train], minlength=len(e) + 1) / train.sum()
        c_own = np.mean(count_bin(a["n_view"], e) == y)
        c_clear = np.mean(count_bin(a["n_view_clear"], e) == y)
        lines.append(f"  {len(e) + 1:2d}   {str(e.tolist()):30s} {np.round(share, 2).tolist()!s:30s}  {c_own:.2f} | {c_clear:.2f}")
    lines += ["",
              "Task B: label = bin(targets visible in THIS view). Ceiling of a counter that misses occluded targets:"]
    for C in n_bins_list:
        e = quantile_edges(a["n_view"][train], C)
        y = count_bin(a["n_view"], e)
        lines.append(f"  {len(e) + 1:2d}   edges {e.tolist()}   ceiling clear-only {np.mean(count_bin(a['n_view_clear'], e) == y):.2f}")
    return "\n".join(lines)


def assign_splits(rows: list[dict], heldout_frac: float = 0.2, seed: int = 0) -> list[dict]:
    """MDMT train -> train/heldout by sequence; val -> cal; test -> test. Adds key 'our_split'."""
    train_seqs = sorted({r["seq"] for r in rows if r["split"] == "train"}, key=int)
    rng = np.random.default_rng(seed)
    n_hold = max(1, int(round(heldout_frac * len(train_seqs)))) if train_seqs else 0
    held = set(rng.permutation(train_seqs)[:n_hold].tolist()) if train_seqs else set()
    for r in rows:
        r["our_split"] = {"val": "cal", "test": "test"}.get(r["split"], "heldout" if r["seq"] in held else "train")
    return rows


SCENE_EDGES = np.array([45, 76, 108])
NUM_CLASSES = len(SCENE_EDGES) + 1
FRAME_FIELDS = ["our_split", "seq", "view", "frame", "scene_id", "label", "n_union", "n_view", "n_view_clear",
                "image", "source"]


def frame_rows(index_rows: list[dict], heldout_frac: float = 0.2, seed: int = 0) -> list[dict]:
    """Index rows -> training rows with label, split and scene id (both views of a time instant share it)."""
    rows = assign_splits([dict(r) for r in index_rows], heldout_frac, seed)
    out = []
    for r in rows:
        out.append({"our_split": r["our_split"], "seq": r["seq"], "view": r["view"], "frame": r["frame"],
                    "scene_id": f"{r['seq']}_{r['frame']:06d}", "label": int(count_bin(r["n_union"], SCENE_EDGES)),
                    "n_union": r["n_union"], "n_view": r["n_view"], "n_view_clear": r["n_view_clear"],
                    "image": str(Path(r["image"]).with_suffix(".jpg").as_posix()), "source": r["image"]})
    return out


def write_frames(rows: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FRAME_FIELDS)
        w.writeheader()
        w.writerows(rows)


def read_frames(path: str | Path) -> list[dict]:
    ints = {"view", "frame", "label", "n_union", "n_view", "n_view_clear"}
    with open(path, newline="", encoding="utf-8") as fh:
        return [{k: (int(v) if k in ints else v) for k, v in r.items()} for r in csv.DictReader(fh)]


def _resize_one(job):
    src, dst, w, h = job
    if Path(dst).exists():
        return 0
    from PIL import Image

    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im.convert("RGB").resize((w, h), Image.BILINEAR).save(dst, quality=92)
    return 1


def resize_images(rows: list[dict], root: str | Path, img_root: str | Path, w: int = 512, h: int = 288,
                  workers: int = 8) -> int:
    """Pre-resize every frame once (1920x1080 -> w x h JPEG) so training reads small files. Restartable:
    existing outputs are skipped. Returns the number of images written."""
    jobs = [(str(Path(root) / r["source"]), str(Path(img_root) / r["image"]), w, h) for r in rows]
    if workers <= 1:
        return sum(_resize_one(j) for j in jobs)
    from multiprocessing import Pool

    done = 0
    with Pool(workers) as pool:
        for i, n in enumerate(pool.imap_unordered(_resize_one, jobs, chunksize=64), 1):
            done += n
            if i % 2000 == 0:
                print(f"[mdmt] resized {i}/{len(jobs)}", flush=True)
    return done


GRID_H, GRID_W = 36, 64


def _frame_size(folder: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        first = next(folder.glob("*.jpg"))
        with Image.open(first) as im:
            return im.size
    except Exception:
        return 1920, 1080


def grid_pair(pair: SeqPair, gh: int = GRID_H, gw: int = GRID_W) -> dict[tuple[int, int], np.ndarray]:
    """{(view index, image number): (gh, gw) count of this view's visible targets per cell} for frames present
    in both views (same alignment rule as index_pair)."""
    tracks = {v: parse_tracks(pair.xmls[v], centers=True) for v in VIEWS}
    nums = {v: image_numbers(pair.dirs[v]) for v in VIEWS}
    common = sorted(set(nums["1"]) & set(nums["2"]))
    out = {}
    for v in VIEWS:
        if not common:
            break
        off, _ = best_offset(common, set(tracks[v]))
        W, H = _frame_size(pair.dirs[v])
        for n in common:
            g = np.zeros((gh, gw), dtype=np.uint16)
            for _, _, cx, cy in tracks[v].get(n - off, {}).values():
                g[min(gh - 1, max(0, int(cy / H * gh))), min(gw - 1, max(0, int(cx / W * gw)))] += 1
            out[(int(v) - 1, n)] = g
    return out


def build_grids(root: str | Path, rows: list[dict], gh: int = GRID_H, gw: int = GRID_W) -> np.ndarray:
    """Count grids aligned with ``rows`` (frames.csv order): density supervision for SPCRNet."""
    want = {(r["seq"], r["view"], r["frame"]): i for i, r in enumerate(rows)}
    grids = np.zeros((len(rows), gh, gw), dtype=np.uint16)
    seqs = {r["seq"] for r in rows}
    found = 0
    for p in find_pairs(root):
        if p.seq not in seqs:
            continue
        for (v, n), g in grid_pair(p, gh, gw).items():
            i = want.get((p.seq, v, n))
            if i is not None:
                grids[i] = g
                found += 1
        print(f"[mdmt] grids: sequence {p.seq} done", flush=True)
    if found != len(rows):
        print(f"[mdmt] WARNING: grids found for {found} of {len(rows)} rows")
    return grids


def save_grids(grids: np.ndarray, rows: list[dict], path: str | Path) -> None:
    if grids.max() < 256:
        grids = grids.astype(np.uint8)
    np.savez_compressed(path, grid=grids, seq=np.array([r["seq"] for r in rows]),
                        view=np.array([r["view"] for r in rows]), frame=np.array([r["frame"] for r in rows]))


def load_grids(path: str | Path) -> dict[tuple[str, int, int], np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {(str(s), int(v), int(f)): g for s, v, f, g in zip(z["seq"], z["view"], z["frame"], z["grid"])}


DET_CLASSES = ("person", "car", "other")          # detector labels 1..3 (0 = background)


def boxes_pair(pair: SeqPair) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """{(view index, image number): (boxes (n,4) xyxy normalised to [0,1], class ids 1..3)} for frames present in
    both views (same alignment rule as index_pair)."""
    tracks = {v: parse_tracks(pair.xmls[v], boxes=True) for v in VIEWS}
    nums = {v: image_numbers(pair.dirs[v]) for v in VIEWS}
    common = sorted(set(nums["1"]) & set(nums["2"]))
    out = {}
    for v in VIEWS:
        if not common:
            break
        off, _ = best_offset(common, set(tracks[v]))
        W, H = _frame_size(pair.dirs[v])
        for n in common:
            vals = list(tracks[v].get(n - off, {}).values())
            b = np.array([[x0 / W, y0 / H, x1 / W, y1 / H] for _, _, x0, y0, x1, y1 in vals], np.float32).reshape(-1, 4)
            c = np.array([DET_CLASSES.index(_cls(lab)) + 1 for lab, *_ in vals], np.uint8)
            out[(int(v) - 1, n)] = (np.clip(b, 0.0, 1.0), c)
    return out


def build_boxes(root: str | Path, rows: list[dict]) -> dict[str, np.ndarray]:
    """Ragged arrays aligned with ``rows``: boxes (N,4), cls (N,), ptr (S+1,) with row i -> [ptr[i], ptr[i+1])."""
    want = {(r["seq"], r["view"], r["frame"]): i for i, r in enumerate(rows)}
    per_row: list = [None] * len(rows)
    seqs = {r["seq"] for r in rows}
    for p in find_pairs(root):
        if p.seq not in seqs:
            continue
        for (v, n), bc in boxes_pair(p).items():
            i = want.get((p.seq, v, n))
            if i is not None:
                per_row[i] = bc
        print(f"[mdmt] boxes: sequence {p.seq} done", flush=True)
    empty = (np.zeros((0, 4), np.float32), np.zeros(0, np.uint8))
    per_row = [x if x is not None else empty for x in per_row]
    ptr = np.concatenate([[0], np.cumsum([len(c) for _, c in per_row])]).astype(np.int64)
    return {"boxes": np.concatenate([b for b, _ in per_row]).astype(np.float32),
            "cls": np.concatenate([c for _, c in per_row]).astype(np.uint8), "ptr": ptr,
            "seq": np.array([r["seq"] for r in rows]), "view": np.array([r["view"] for r in rows]),
            "frame": np.array([r["frame"] for r in rows])}


def load_boxes(path: str | Path) -> dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray]]:
    with np.load(path, allow_pickle=False) as z:
        b, c, ptr = z["boxes"], z["cls"], z["ptr"]
        return {(str(s), int(v), int(f)): (b[ptr[i]:ptr[i + 1]], c[ptr[i]:ptr[i + 1]])
                for i, (s, v, f) in enumerate(zip(z["seq"], z["view"], z["frame"]))}


#observability ceilings
def ceilings(rows: list[dict], fit_split: str = "train", eval_split: str = "test") -> dict:
    """How well can the scene level be predicted from PERFECT per-view counts? Log-linear maps fitted on
    ``fit_split`` and scored on ``eval_split`` (MDMT splits). This bounds every model that sees one view (LOCAL
    and A2G) and shows what a second view adds (A2A)."""
    key = {(r["seq"], r["frame"], r["view"]): r for r in rows}
    feats = {"own count (all annotated)": [], "own count (non-occluded only)": [],
             "both views (all annotated)": [], "both views (non-occluded only)": []}
    y, sp, t = [], [], []
    for r in rows:
        o = key.get((r["seq"], r["frame"], 1 - r["view"]))
        if o is None:
            continue
        a1, a2 = np.log1p(r["n_view"]), np.log1p(o["n_view"])
        c1, c2 = np.log1p(r["n_view_clear"]), np.log1p(o["n_view_clear"])
        feats["own count (all annotated)"].append([a1, 1.0])
        feats["own count (non-occluded only)"].append([c1, 1.0])
        feats["both views (all annotated)"].append([a1, a2, 1.0])
        feats["both views (non-occluded only)"].append([c1, c2, 1.0])
        y.append(int(count_bin(r["n_union"], SCENE_EDGES)))
        t.append(np.log1p(r["n_union"]))
        sp.append(r["split"])
    y, t, sp = np.array(y), np.array(t), np.array(sp)
    fit, ev = sp == fit_split, sp == eval_split
    out = {}
    for name, X in feats.items():
        X = np.array(X)
        w, *_ = np.linalg.lstsq(X[fit], t[fit], rcond=None)
        pred = count_bin(np.expm1(X[ev] @ w), SCENE_EDGES)
        out[name] = float(np.mean(pred == y[ev]))
    return out


def split_summary(rows: list[dict]) -> str:
    lines = ["split      rows  seqs  class shares"]
    for sp in ("train", "heldout", "cal", "test"):
        rs = [r for r in rows if r["our_split"] == sp]
        if not rs:
            continue
        sh = np.bincount([r["label"] for r in rs], minlength=NUM_CLASSES) / len(rs)
        lines.append(f"{sp:8s} {len(rs):6d} {len({r['seq'] for r in rs}):5d}  {np.round(sh, 2).tolist()}")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["stats", "prepare", "grids", "boxes", "ceiling"],
                    help="stats: index + label report; prepare: labels, splits, resized images -> frames.csv "
                         "(+ grids); grids: per-cell target counts for density supervision -> grids.npz")
    ap.add_argument("--root", default=None, help="MDMT-VisDrone folder (not needed for 'ceiling')")
    ap.add_argument("--out", default="caches/mdmt")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=288)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--img_dir", default="img", help="sub-folder of --out for the resized images (e.g. img1024)")
    ap.add_argument("--grid_h", type=int, default=GRID_H)
    ap.add_argument("--grid_w", type=int, default=GRID_W)
    args = ap.parse_args(argv)
    out = Path(args.out)
    if args.command != "ceiling" and not args.root:
        raise SystemExit(f"--root <MDMT-VisDrone folder> is required for '{args.command}'")
    gname = f"grids_{args.grid_h}x{args.grid_w}.npz"
    if args.command == "ceiling":
        rows = read_index(out / "index.csv")
        res = ceilings(rows)
        print("Accuracy on the MDMT test split of the best log-linear map from PERFECT counts to the scene level:")
        for k, v in res.items():
            print(f"  {k:32s} {v:.3f}")
        (out / "ceilings.json").write_text(__import__("json").dumps(res, indent=2), encoding="utf-8")
        return
    if args.command == "boxes":
        rows = read_frames(out / "frames.csv")
        bx = build_boxes(args.root, rows)
        np.savez_compressed(out / "boxes.npz", **bx)
        n = np.diff(bx["ptr"])
        nv = np.array([r["n_view"] for r in rows])
        print(f"wrote {out / 'boxes.npz'}: {len(bx['cls'])} boxes; per-row counts match n_view on {np.mean(n == nv):.1%}")
        return
    if args.command == "grids":
        rows = read_frames(out / "frames.csv")
        g = build_grids(args.root, rows, args.grid_h, args.grid_w)
        save_grids(g, rows, out / gname)
        tot = g.reshape(len(rows), -1).sum(1)
        nv = np.array([r["n_view"] for r in rows])
        print(f"wrote {out / gname}: {g.shape}; grid totals match n_view on {np.mean(tot == nv):.1%} of rows")
        return
    if args.command == "prepare":
        idx = out / "index.csv"
        if not idx.exists():
            raise SystemExit(f"{idx} not found: run the 'stats' command first")
        rows = frame_rows(read_index(idx))
        write_frames(rows, out / "frames.csv")
        print(split_summary(rows), flush=True)
        n = resize_images(rows, args.root, out / args.img_dir, args.width, args.height, args.workers)
        if not (out / gname).exists():
            save_grids(build_grids(args.root, rows, args.grid_h, args.grid_w), rows, out / gname)
        print(f"wrote {out / 'frames.csv'} and {gname}; resized {n} new image(s) into {out / args.img_dir}")
        return
    rows, recs = build_index(args.root)
    if not rows:
        raise SystemExit("no paired frames found; check --root")
    write_index(rows, out / "index.csv")
    bad = [r for r in recs if min(r.get("coverage_v1", 0), r.get("coverage_v2", 0)) < 0.9]
    report = [f"MDMT root: {Path(args.root).resolve()}",
              f"sequence pairs: {len(recs)}  ({dict(Counter(r['split'] for r in recs))})",
              f"frame alignment offsets used: {dict(Counter((r.get('offset_v1'), r.get('offset_v2')) for r in recs))}",
              f"pairs with xml coverage < 90%: {[(r['seq'], round(r['coverage_v1'], 2), round(r['coverage_v2'], 2)) for r in bad]}",
              "", label_report(rows)]
    text = "\n".join(report)
    (out / "label_report.txt").write_text(text, encoding="utf-8")
    print("\n" + text + f"\n\nwrote {out / 'index.csv'} and {out / 'label_report.txt'}")


if __name__ == "__main__":
    main()
