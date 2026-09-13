#!/usr/bin/env python3
"""
scripts/prepare_resized_data.py

Verkleinert die (sehr großen) Original-Aufnahmen EINMALIG auf die Trainingsauflösung
und legt sie in einem separaten Zielordner ab. Die Originale bleiben unberührt.

Hintergrund: Die Kameras liefern 4096x3000-PNGs (~17 MB). Diese pro Epoche neu zu
dekodieren ist der Flaschenhals des Trainings. Nach dem Verkleinern lädt der DataLoader
ein Vielfaches schneller, ohne dass sich die Modell-Eingabe ändert (die Pipeline skaliert
ohnehin auf image_size x image_size).

--size nimmt eine Zahl (quadratisch) oder BREITExHOEHE. Nicht-quadratisch lohnt sich
bei nicht-quadratischen Bildern: ein 2999x1200-Zuschnitt auf 512x512 gequetscht
verliert horizontal 5,9x, auf 800x320 dagegen gleichmaessig 3,75x - bei praktisch
gleicher Pixelzahl, also gleicher Rechenzeit.

Beispiel:
    python scripts/prepare_resized_data.py                       # data -> data_512, 512px
    python scripts/prepare_resized_data.py --size 512 --target data_512
    python scripts/prepare_resized_data.py --source data --target data_384 --size 384
    python scripts/prepare_resized_data.py --source dataStufe2zugeschnitten \
        --target dataStufe2zugeschnitten_800x320 --size 800x320    # BREITExHOEHE
    python scripts/prepare_resized_data.py --mirror              # Ziel exakt spiegeln (verwaiste Bilder loeschen)
"""
import argparse
import os
import sys

from PIL import Image

SPLITS = ["train", "val", "test"]
CLASSES = ["io", "nio"]
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def parse_groesse(text):
    """'512' -> (512, 512), '800x320' -> (800, 320), jeweils als (Breite, Hoehe) fuer PIL."""
    teile = str(text).lower().replace("*", "x").split("x")
    try:
        werte = [int(t) for t in teile]
    except ValueError:
        raise argparse.ArgumentTypeError(f"--size erwartet '512' oder '800x320', bekam {text!r}")
    if len(werte) == 1:
        werte *= 2
    if len(werte) != 2 or any(w <= 0 for w in werte):
        raise argparse.ArgumentTypeError(f"--size erwartet '512' oder '800x320', bekam {text!r}")
    return werte[0], werte[1]


def parse_args():
    p = argparse.ArgumentParser(description="Trainingsbilder vorab auf image_size verkleinern.")
    p.add_argument("--source", default="data", help="Quellordner mit train/val/test (default: data)")
    p.add_argument("--target", default="data_512", help="Zielordner (default: data_512)")
    p.add_argument("--size", type=parse_groesse, default="512",
                   help="Zielgroesse: Zahl fuer quadratisch oder BREITExHOEHE, z.B. 800x320 (default: 512)")
    p.add_argument("--force", action="store_true", help="Vorhandene Zielbilder überschreiben")
    p.add_argument("--mirror", action="store_true",
                   help="Ziel exakt spiegeln: Bilder, die im Ziel liegen, aber in der Quelle "
                        "fehlen (z.B. gelöscht/verschoben), werden aus dem Ziel entfernt")
    return p.parse_args()


def remove_orphans(dst_dir, src_names):
    """Löscht Bilddateien in dst_dir, deren Name nicht in src_names vorkommt.

    Gibt die Anzahl der entfernten Dateien zurück. Nur Bilddateien werden
    angefasst, damit Fremddateien (z.B. Notizen) im Ziel unberührt bleiben.
    """
    removed = 0
    for fname in os.listdir(dst_dir):
        if os.path.splitext(fname)[1].lower() not in IMG_EXTENSIONS:
            continue
        if fname not in src_names:
            os.remove(os.path.join(dst_dir, fname))
            removed += 1
    return removed


def main():
    args = parse_args()
    size = args.size                      # (Breite, Hoehe) - so erwartet es PIL.resize
    breite, hoehe = size

    if not os.path.isdir(args.source):
        print(f"[FEHLER] Quellordner nicht gefunden: {args.source}")
        sys.exit(1)
    if os.path.abspath(args.source) == os.path.abspath(args.target):
        print("[FEHLER] Quelle und Ziel dürfen nicht identisch sein (Originale würden überschrieben).")
        sys.exit(1)

    total, converted, skipped, removed = 0, 0, 0, 0
    for split in SPLITS:
        for cls in CLASSES:
            src_dir = os.path.join(args.source, split, cls)
            dst_dir = os.path.join(args.target, split, cls)

            if not os.path.isdir(src_dir):
                # Quelle fehlt: im Mirror-Modus das komplette Ziel-Pendant leeren
                if args.mirror and os.path.isdir(dst_dir):
                    removed += remove_orphans(dst_dir, set())
                continue
            os.makedirs(dst_dir, exist_ok=True)

            src_names = set()
            for fname in sorted(os.listdir(src_dir)):
                if os.path.splitext(fname)[1].lower() not in IMG_EXTENSIONS:
                    continue
                src_names.add(fname)
                total += 1
                src_path = os.path.join(src_dir, fname)
                dst_path = os.path.join(dst_dir, fname)
                if os.path.exists(dst_path) and not args.force:
                    skipped += 1
                    continue
                try:
                    with Image.open(src_path) as im:
                        # Bilinear = identisches Verfahren wie torchvision T.Resize (Default)
                        im.convert("RGB").resize(size, Image.BILINEAR).save(dst_path)
                    converted += 1
                except Exception as exc:
                    print(f"  [WARN] übersprungen (Lesefehler): {src_path} ({exc})")

            # Mirror: Zielbilder ohne Quell-Pendant (gelöscht/verschoben) entfernen
            if args.mirror:
                removed += remove_orphans(dst_dir, src_names)

            n = len([f for f in os.listdir(dst_dir)
                     if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS])
            print(f"  {split:5s}/{cls:3s}: {n} Bilder in {dst_dir}")

    print("-" * 50)
    print(f"Fertig: {converted} konvertiert, {skipped} übersprungen (bereits vorhanden), "
          f"{total} gesamt -> Zielauflösung {breite}x{hoehe} px (BxH)")
    if args.mirror:
        print(f"Mirror: {removed} verwaiste Zielbilder entfernt (nicht mehr in {args.source})")
    print(f"Nächster Schritt: in configs/default.yaml 'data_dir: {args.target}' und "
          f"'image_size: {hoehe}' setzen." if breite == hoehe else
          f"Nächster Schritt: in configs/default.yaml 'data_dir: {args.target}' und "
          f"'image_size: [{hoehe}, {breite}]' setzen  (Reihenfolge [Hoehe, Breite]).")


if __name__ == "__main__":
    main()
