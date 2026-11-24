"""
pipeline/download_extract.py
Downloads and extracts historical Delhi bus GPS SQLite databases from the 7z archive.
Supports streaming extraction via HTTP range requests or full download, preserving disk space.
"""

import os
import sys
import io
import argparse
import urllib.request
import py7zr

ARCHIVE_URL = "https://pub-091bd9e794bb4877b10b29e6df80f065.r2.dev/2026-10/delhi-dtc-bus-gtfs-rt-2026-10.7z"
RAW_DIR = os.path.join("data", "raw")


class HttpRangeFile(io.RawIOBase):
    """Seekable file-like object over HTTP range requests."""

    def __init__(self, url: str):
        self.url = url
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            self.size = int(resp.headers["Content-Length"])
        self.pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def readinto(self, b) -> int:
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(b) - 1, self.size - 1)
        req = urllib.request.Request(
            self.url,
            headers={"User-Agent": "Mozilla/5.0", "Range": f"bytes={self.pos}-{end}"},
        )
        with urllib.request.urlopen(req) as resp:
            chunk = resp.read()
        n = len(chunk)
        b[:n] = chunk
        self.pos += n
        return n


def list_archive_contents():
    """Returns the list of database file names inside the archive."""
    print("Fetching archive manifest...")
    f = io.BufferedReader(HttpRangeFile(ARCHIVE_URL), buffer_size=4 * 1024 * 1024)
    archive = py7zr.SevenZipFile(f, mode="r")
    return archive.getnames()


def extract_days(days: list[int], output_dir: str = RAW_DIR, force: bool = False):
    """
    Extracts specific days (1 to 31) into output_dir.
    Skips if file already exists unless force=True.
    """
    os.makedirs(output_dir, exist_ok=True)
    target_names = [f"bus_movements_2026_10_{d:02d}.db" for d in days]

    # Filter out already extracted files
    needed = []
    for name in target_names:
        dest_path = os.path.join(output_dir, name)
        if os.path.exists(dest_path) and not force:
            print(f"File already exists: {dest_path} (skipping)")
        else:
            needed.append(name)

    if not needed:
        print("All requested files already exist in destination.")
        return

    print(f"Extracting {len(needed)} file(s) from remote archive: {needed}...")
    f = io.BufferedReader(HttpRangeFile(ARCHIVE_URL), buffer_size=16 * 1024 * 1024)
    archive = py7zr.SevenZipFile(f, mode="r")

    for target in needed:
        dest_path = os.path.join(output_dir, target)
        print(f"Extracting {target} -> {dest_path}...")
        archive.extract(path=output_dir, targets=[target])
        if os.path.exists(dest_path):
            size_mb = os.path.getsize(dest_path) / (1024 * 1024)
            print(f"Successfully extracted {target} ({size_mb:.1f} MB)")
        else:
            print(f"Warning: Extraction did not create {dest_path}")


def main():
    parser = argparse.ArgumentParser(description="Download and extract Delhi bus GPS SQLite databases.")
    parser.add_argument("--days", type=str, default="1", help="Comma-separated day numbers or ranges (e.g. '1', '1,2,3', '1-5')")
    parser.add_argument("--all", action="store_true", help="Extract all 31 days (warning: requires ~115 GB if all kept uncompressed)")
    parser.add_argument("--output", type=str, default=RAW_DIR, help="Destination directory for raw SQLite databases")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument("--list", action="store_true", help="List files in archive and exit")

    args = parser.parse_args()

    if args.list:
        files = list_archive_contents()
        print(f"Found {len(files)} files:")
        for name in files:
            print(f"  {name}")
        return

    days_to_extract = set()
    if args.all:
        days_to_extract = set(range(1, 32))
    else:
        for part in args.days.split(","):
            part = part.strip()
            if "-" in part:
                start, end = map(int, part.split("-"))
                days_to_extract.update(range(start, end + 1))
            else:
                days_to_extract.add(int(part))

    days = sorted(list(days_to_extract))
    print(f"Target days to extract: {days}")
    extract_days(days, output_dir=args.output, force=args.force)


if __name__ == "__main__":
    main()
