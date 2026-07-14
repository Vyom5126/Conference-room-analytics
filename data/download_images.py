"""
download_images.py
──────────────────
Downloads a curated set of conference room images from public Unsplash/
Wikimedia sources into data/images/.

Targets 60–80 images covering a spectrum from very clean to very cluttered
conference rooms — enough to test the full pipeline end-to-end.

Usage:
    python data/download_images.py [--output-dir PATH] [--max N]
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

# ── Curated image URLs ────────────────────────────────────────────────────────
# Mix of clean/organised and messy/cluttered conference rooms from
# Unsplash (free to use), Wikimedia Commons (CC), and Pexels (free license).
# Each entry: (url, local_filename)
IMAGE_SOURCES: list[tuple[str, str]] = [
    # ── Clean / Professional rooms ──────────────────────────────────────────
    ("https://images.unsplash.com/photo-1497366216548-37526070297c?w=1200&q=80", "clean_room_01.jpg"),
    ("https://images.unsplash.com/photo-1497366811353-6870744d04b2?w=1200&q=80", "clean_room_02.jpg"),
    ("https://images.unsplash.com/photo-1497366754035-f200968a6e72?w=1200&q=80", "clean_room_03.jpg"),
    ("https://images.unsplash.com/photo-1431540015161-0bf868a2d407?w=1200&q=80", "clean_room_04.jpg"),
    ("https://images.unsplash.com/photo-1505409859467-3a796fd5798e?w=1200&q=80", "clean_room_05.jpg"),
    ("https://images.unsplash.com/photo-1504384308090-c894fdcc538d?w=1200&q=80", "clean_room_06.jpg"),
    ("https://images.unsplash.com/photo-1519389950473-47ba0277781c?w=1200&q=80", "clean_room_07.jpg"),
    ("https://images.unsplash.com/photo-1542744173-8e7e53415bb0?w=1200&q=80", "clean_room_08.jpg"),
    ("https://images.unsplash.com/photo-1568992687947-868a62a9f521?w=1200&q=80", "clean_room_09.jpg"),
    ("https://images.unsplash.com/photo-1556761175-5973dc0f32e7?w=1200&q=80", "clean_room_10.jpg"),
    ("https://images.unsplash.com/photo-1573165231977-3f0e27806045?w=1200&q=80", "clean_room_11.jpg"),
    ("https://images.unsplash.com/photo-1581578731548-c64695cc6952?w=1200&q=80", "clean_room_12.jpg"),
    ("https://images.unsplash.com/photo-1587825140708-dfaf72ae4b04?w=1200&q=80", "clean_room_13.jpg"),
    ("https://images.unsplash.com/photo-1533750349088-cd871a92f312?w=1200&q=80", "clean_room_14.jpg"),
    ("https://images.unsplash.com/photo-1600880292203-757bb62b4baf?w=1200&q=80", "clean_room_15.jpg"),
    ("https://images.unsplash.com/photo-1560179707-f14e90ef3623?w=1200&q=80", "clean_room_16.jpg"),
    ("https://images.unsplash.com/photo-1486325212027-8081e485255e?w=1200&q=80", "clean_room_17.jpg"),
    ("https://images.unsplash.com/photo-1524758631624-e2822e304c36?w=1200&q=80", "clean_room_18.jpg"),
    ("https://images.unsplash.com/photo-1612532869120-04c5e2c18df2?w=1200&q=80", "clean_room_19.jpg"),
    ("https://images.unsplash.com/photo-1553028826-f4804a6dba3b?w=1200&q=80", "clean_room_20.jpg"),

    # ── Moderately busy / semi-organised ───────────────────────────────────
    ("https://images.unsplash.com/photo-1521737852567-6949f3f9f2b5?w=1200&q=80", "medium_room_01.jpg"),
    ("https://images.unsplash.com/photo-1522071820081-009f0129c71c?w=1200&q=80", "medium_room_02.jpg"),
    ("https://images.unsplash.com/photo-1525130413817-d45c1d127c42?w=1200&q=80", "medium_room_03.jpg"),
    ("https://images.unsplash.com/photo-1552664730-d307ca884978?w=1200&q=80", "medium_room_04.jpg"),
    ("https://images.unsplash.com/photo-1517245386807-bb43f82c33c4?w=1200&q=80", "medium_room_05.jpg"),
    ("https://images.unsplash.com/photo-1557804506-669a67965ba0?w=1200&q=80", "medium_room_06.jpg"),
    ("https://images.unsplash.com/photo-1516321318423-f06f85e504b3?w=1200&q=80", "medium_room_07.jpg"),
    ("https://images.unsplash.com/photo-1575505586569-646b2ca898fc?w=1200&q=80", "medium_room_08.jpg"),
    ("https://images.unsplash.com/photo-1578574577315-3fbeb0cecdc2?w=1200&q=80", "medium_room_09.jpg"),
    ("https://images.unsplash.com/photo-1543269664-56d93c1b41a6?w=1200&q=80", "medium_room_10.jpg"),
    ("https://images.unsplash.com/photo-1603201667141-5a2d4c673378?w=1200&q=80", "medium_room_11.jpg"),
    ("https://images.unsplash.com/photo-1562564055-71e051d33c19?w=1200&q=80", "medium_room_12.jpg"),
    ("https://images.unsplash.com/photo-1542744095-fcf48d80b0fd?w=1200&q=80", "medium_room_13.jpg"),
    ("https://images.unsplash.com/photo-1551836022-d5d88e9218df?w=1200&q=80", "medium_room_14.jpg"),
    ("https://images.unsplash.com/photo-1593642632559-0c6d3fc62b89?w=1200&q=80", "medium_room_15.jpg"),
    ("https://images.unsplash.com/photo-1590402494682-cd3fb53b1f70?w=1200&q=80", "medium_room_16.jpg"),
    ("https://images.unsplash.com/photo-1588196749597-9ff075ee6b5b?w=1200&q=80", "medium_room_17.jpg"),
    ("https://images.unsplash.com/photo-1565301660306-29e08751cc53?w=1200&q=80", "medium_room_18.jpg"),
    ("https://images.unsplash.com/photo-1581090464777-f3220bbe1b8b?w=1200&q=80", "medium_room_19.jpg"),
    ("https://images.unsplash.com/photo-1556742400-b5b7c512047a?w=1200&q=80", "medium_room_20.jpg"),

    # ── Active use / occupied ───────────────────────────────────────────────
    ("https://images.unsplash.com/photo-1556761175-b413da4baf72?w=1200&q=80", "occupied_room_01.jpg"),
    ("https://images.unsplash.com/photo-1517048676732-d65bc937f952?w=1200&q=80", "occupied_room_02.jpg"),
    ("https://images.unsplash.com/photo-1450101499163-c8848c66ca85?w=1200&q=80", "occupied_room_03.jpg"),
    ("https://images.unsplash.com/photo-1460925895917-afdab827c52f?w=1200&q=80", "occupied_room_04.jpg"),
    ("https://images.unsplash.com/photo-1455849318743-b2233052fcff?w=1200&q=80", "occupied_room_05.jpg"),
    ("https://images.unsplash.com/photo-1532619675605-1ede6c2ed2b0?w=1200&q=80", "occupied_room_06.jpg"),
    ("https://images.unsplash.com/photo-1454165804606-c3d57bc86b40?w=1200&q=80", "occupied_room_07.jpg"),
    ("https://images.unsplash.com/photo-1565008576549-57b1ffe9de94?w=1200&q=80", "occupied_room_08.jpg"),
    ("https://images.unsplash.com/photo-1535016120720-40c646be5580?w=1200&q=80", "occupied_room_09.jpg"),
    ("https://images.unsplash.com/photo-1504868584819-f8e8b4b6d7e3?w=1200&q=80", "occupied_room_10.jpg"),

    # ── Messy / post-meeting ────────────────────────────────────────────────
    ("https://images.unsplash.com/photo-1572021335469-31706a17aaef?w=1200&q=80", "messy_room_01.jpg"),
    ("https://images.unsplash.com/photo-1588196749597-9ff075ee6b5b?w=1200&q=80", "messy_room_02.jpg"),
    ("https://images.unsplash.com/photo-1610416900944-50e9bfb73770?w=1200&q=80", "messy_room_03.jpg"),
    ("https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1200&q=80", "messy_room_04.jpg"),
    ("https://images.unsplash.com/photo-1581091226825-a6a2a5aee158?w=1200&q=80", "messy_room_05.jpg"),
    ("https://images.unsplash.com/photo-1568992687947-868a62a9f521?w=1200&q=80", "messy_room_06.jpg"),
    ("https://images.unsplash.com/photo-1531545514256-b1400bc00f31?w=1200&q=80", "messy_room_07.jpg"),
    ("https://images.unsplash.com/photo-1596462502278-27bfdc403348?w=1200&q=80", "messy_room_08.jpg"),
    ("https://images.unsplash.com/photo-1606857521015-7f9fcf423740?w=1200&q=80", "messy_room_09.jpg"),
    ("https://images.unsplash.com/photo-1620330488787-ff0398adaa6c?w=1200&q=80", "messy_room_10.jpg"),
]


def download_image(
    url: str,
    dest: Path,
    timeout: int = 15,
    retries: int = 3,
) -> bool:
    """Download a single image with retry logic. Returns True on success."""
    if dest.exists():
        return True  # already downloaded

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=timeout, stream=True)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if "image" not in content_type and attempt == retries:
                log.debug("Non-image content-type: %s for %s", content_type, url)

            with open(dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Verify it's a valid image
            from PIL import Image
            try:
                with Image.open(dest) as img:
                    img.verify()
            except Exception:
                dest.unlink(missing_ok=True)
                return False

            return True

        except requests.RequestException as e:
            if attempt < retries:
                time.sleep(1.5 * attempt)
            else:
                log.debug("Failed (%d/%d): %s — %s", attempt, retries, url, e)
                return False

    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Download conference room images.")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "data" / "images")
    parser.add_argument("--max", type=int, default=len(IMAGE_SOURCES),
                        help="Maximum number of images to download")
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = IMAGE_SOURCES[: args.max]
    log.info("Downloading %d images → %s", len(sources), output_dir)

    success = 0
    failed = 0
    skipped = 0

    for url, filename in tqdm(sources, desc="Downloading"):
        dest = output_dir / filename
        if dest.exists():
            skipped += 1
            continue
        ok = download_image(url, dest)
        if ok:
            success += 1
        else:
            failed += 1
        time.sleep(0.1)  # polite rate limit

    log.info(
        "Done — ✓ %d downloaded  |  ↷ %d skipped  |  ✗ %d failed",
        success, skipped, failed,
    )

    # List what we have
    imgs = list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.png"))
    log.info("Total images available: %d", len(imgs))

    if len(imgs) < 10:
        log.warning(
            "Only %d images found. Some URLs may have expired. "
            "Please add your own images to %s",
            len(imgs), output_dir,
        )
    else:
        log.info(
            "✓ Ready! Run 'python data/generate_labels.py' to create labels.csv"
        )


if __name__ == "__main__":
    main()
