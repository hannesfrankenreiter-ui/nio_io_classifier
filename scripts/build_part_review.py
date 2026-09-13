"""
scripts/build_part_review.py

Rekonstruktion der Zuordnung Bild -> physisches Bauteil (4 Bauteile je Klasse).

Die Dateinamen enthalten nur Kamera-Zeitstempel, keine Bauteil-ID. Dieses Skript
zerlegt alle Aufnahmen (train+val+test zusammen) pro Klasse in zusammenhaengende
Aufnahme-Segmente (Luecke > Schwelle = neues Segment) und erzeugt:

  outputs/part_review/segments.csv          -> ein Segment pro Zeile, Spalte
                                               'bauteil_id' zum manuellen Ausfuellen
  outputs/part_review/segments_review.html  -> Beispielbilder je Segment zum
                                               visuellen Zuordnen (im Browser oeffnen)

Danach kann aus segments.csv ein Bauteil-Level-Split (Leave-One-Part-Out)
abgeleitet werden.

Aufruf:
    python scripts/build_part_review.py [--threshold 30]
"""
import argparse
import csv
import re
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data_512"
OUT_DIR = PROJECT_ROOT / "outputs" / "part_review"

FILENAME_PATTERN = re.compile(r"sick_midicam2_(\d{8})_(\d{6})_(\d{3})\.png")
SPLITS = ["train", "val", "test"]
CLASSES = ["io", "nio"]


def collect_images(cls: str):
    """Alle Bilder einer Klasse ueber alle Splits, sortiert nach Zeitstempel."""
    images = []
    for split in SPLITS:
        for f in sorted((DATA_DIR / split / cls).glob("*.png")):
            m = FILENAME_PATTERN.match(f.name)
            if not m:
                print(f"  WARNUNG: unerwarteter Dateiname, uebersprungen: {split}/{cls}/{f.name}")
                continue
            date_str, time_str, ms = m.groups()
            ts = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S").timestamp() + int(ms) / 1000
            images.append({"ts": ts, "name": f.name, "split": split})
    images.sort(key=lambda x: x["ts"])
    return images


def segment_images(images: list, threshold_s: float):
    """Zerlegt die zeitlich sortierte Bildliste an Luecken > threshold_s."""
    segments = []
    current = [images[0]]
    for prev, img in zip(images, images[1:]):
        if img["ts"] - prev["ts"] > threshold_s:
            segments.append(current)
            current = []
        current.append(img)
    segments.append(current)
    return segments


def sample_indices(n: int, k: int = 4):
    """k gleichmaessig verteilte Indizes (erstes/letztes Bild immer dabei)."""
    if n <= k:
        return list(range(n))
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def fmt_time(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%d.%m. %H:%M:%S")


def main():
    parser = argparse.ArgumentParser(description="Review-Sheet fuer Bauteil-Zuordnung erzeugen")
    parser.add_argument("--threshold", type=float, default=30.0,
                        help="Luecke in Sekunden, ab der ein neues Segment beginnt (Default: 30)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "segments.csv"
    html_path = OUT_DIR / "segments_review.html"

    csv_rows = []
    html_sections = []

    for cls in CLASSES:
        images = collect_images(cls)
        segments = segment_images(images, args.threshold)
        print(f"Klasse {cls}: {len(images)} Bilder -> {len(segments)} Segmente")

        cards = []
        prev_end_ts = None
        for seg_idx, seg in enumerate(segments, start=1):
            seg_id = f"{cls}_{seg_idx:02d}"
            split_counts = {s: sum(1 for img in seg if img["split"] == s) for s in SPLITS}
            gap_prev = seg[0]["ts"] - prev_end_ts if prev_end_ts is not None else None
            prev_end_ts = seg[-1]["ts"]

            csv_rows.append({
                "segment_id": seg_id,
                "klasse": cls,
                "start": fmt_time(seg[0]["ts"]),
                "ende": fmt_time(seg[-1]["ts"]),
                "n_bilder": len(seg),
                "n_train": split_counts["train"],
                "n_val": split_counts["val"],
                "n_test": split_counts["test"],
                "luecke_zuvor_s": f"{gap_prev:.0f}" if gap_prev is not None else "",
                "erste_datei": seg[0]["name"],
                "letzte_datei": seg[-1]["name"],
                "bauteil_id": "",  # manuell ausfuellen, z. B. io1..io4 / nio1..nio4
            })

            thumbs = []
            for i in sample_indices(len(seg)):
                img = seg[i]
                rel = f"../../data_512/{img['split']}/{cls}/{img['name']}"
                thumbs.append(
                    f'<figure><img src="{rel}" loading="lazy">'
                    f"<figcaption>Bild {i + 1}/{len(seg)}<br>{img['name']}</figcaption></figure>"
                )

            gap_info = f"Luecke zum vorherigen Segment: {gap_prev/60:.1f} min" if gap_prev is not None else "erstes Segment"
            cards.append(
                f'<div class="card"><h3>{seg_id}</h3>'
                f"<p>{fmt_time(seg[0]['ts'])} – {fmt_time(seg[-1]['ts'])} &nbsp;|&nbsp; "
                f"{len(seg)} Bilder (train {split_counts['train']} / val {split_counts['val']} / test {split_counts['test']}) "
                f"&nbsp;|&nbsp; {gap_info}</p>"
                f'<div class="thumbs">{"".join(thumbs)}</div></div>'
            )

        html_sections.append(f"<h2>Klasse {cls} ({len(segments)} Segmente)</h2>" + "".join(cards))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()), delimiter=";")
        writer.writeheader()
        writer.writerows(csv_rows)

    html = f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<title>Bauteil-Zuordnung – Segment-Review</title>
<style>
  body {{ font-family: "Segoe UI", sans-serif; margin: 2rem; background: #f4f4f4; }}
  h1 {{ font-size: 1.4rem; }}
  h2 {{ margin-top: 2.5rem; border-bottom: 2px solid #888; padding-bottom: .3rem; }}
  .card {{ background: #fff; border-radius: 8px; padding: 1rem 1.2rem; margin: 1rem 0;
           box-shadow: 0 1px 3px rgba(0,0,0,.15); }}
  .card h3 {{ margin: 0 0 .3rem; font-size: 1.1rem; }}
  .card p {{ margin: 0 0 .8rem; color: #444; font-size: .9rem; }}
  .thumbs {{ display: flex; flex-wrap: wrap; gap: .8rem; }}
  figure {{ margin: 0; }}
  img {{ width: 300px; height: auto; border: 1px solid #ccc; border-radius: 4px; }}
  figcaption {{ font-size: .75rem; color: #666; max-width: 300px; word-break: break-all; }}
</style>
</head>
<body>
<h1>Segment-Review: Welches Bauteil ist auf welchem Aufnahme-Segment?</h1>
<p>Segmentierung bei Aufnahme-Luecken &gt; {args.threshold:.0f} s. Fuer jedes Segment die
Bauteil-ID (z. B. io1–io4, nio1–nio4) in <code>outputs/part_review/segments.csv</code>
(Spalte <code>bauteil_id</code>) eintragen. Falls innerhalb eines Segments doch ein
Bauteilwechsel zu sehen ist: Segment in der CSV notieren, dann wird feiner segmentiert.</p>
{"".join(html_sections)}
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")

    print(f"\nCSV  : {csv_path}")
    print(f"HTML : {html_path}  (im Browser oeffnen)")


if __name__ == "__main__":
    main()
