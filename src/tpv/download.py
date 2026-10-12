"""Fast parallel downloader for PhysioNet EEGBCI dataset via AWS S3 mirror.

Downloads files directly into the directory structure expected by MNE:
    data/MNE-eegbci-data/files/eegmmidb/1.0.0/S{subject:03d}/S{subject:03d}R{run:02d}.edf
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Sequence

import requests
from tqdm import tqdm

from tpv.config import DATA_DIR, N_SUBJECTS

S3_BASE_URL = "https://physionet-open.s3.amazonaws.com/eegmmidb/1.0.0"


def get_edf_path(subject: int, run: int) -> Path:
    """Return the canonical local file path matching MNE's folder structure."""
    return (
        DATA_DIR
        / "MNE-eegbci-data"
        / "files"
        / "eegmmidb"
        / "1.0.0"
        / f"S{subject:03d}"
        / f"S{subject:03d}R{run:02d}.edf"
    )


def download_run(subject: int, run: int, session: requests.Session | None = None) -> tuple[int, int, bool]:
    """Download a single EDF run if not already present.

    Returns:
        (subject, run, was_downloaded): bool is True if downloaded, False if skipped (cached).
    """
    dest = get_edf_path(subject, run)
    if dest.exists() and dest.stat().st_size > 1000:
        return subject, run, False

    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_dest = dest.with_suffix(".edf.part")

    filename = f"S{subject:03d}R{run:02d}.edf"
    url = f"{S3_BASE_URL}/S{subject:03d}/{filename}"

    http = session or requests
    resp = http.get(url, stream=True, timeout=30)
    resp.raise_for_status()

    with open(temp_dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)

    temp_dest.replace(dest)
    return subject, run, True


def download_data(
    subjects: Iterable[int],
    runs: Iterable[int] = range(1, 15),
    max_workers: int = 8,
) -> dict[str, int]:
    """Download specified subjects and runs in parallel using ThreadPoolExecutor.

    Returns:
        Summary dict with counts of 'downloaded', 'cached', and 'total'.
    """
    tasks = [(s, r) for s in subjects for r in runs]
    if not tasks:
        return {"downloaded": 0, "cached": 0, "total": 0}

    stats = {"downloaded": 0, "cached": 0, "total": len(tasks)}

    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(download_run, s, r, session): (s, r)
                for s, r in tasks
            }

            with tqdm(total=len(tasks), desc="Fetching EEG data", unit="file") as pbar:
                for future in as_completed(future_to_task):
                    s, r = future_to_task[future]
                    try:
                        _, _, downloaded = future.result()
                        if downloaded:
                            stats["downloaded"] += 1
                        else:
                            stats["cached"] += 1
                    except Exception as exc:
                        tqdm.write(f"Error downloading S{s:03d}R{r:02d}: {exc}")
                    pbar.update(1)

    return stats


def parse_range(arg: str) -> list[int]:
    """Parse comma/hyphen separated integer list (e.g. '1', '1-5', '1,3,5-7')."""
    result = []
    for part in arg.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            result.extend(range(int(start), int(end) + 1))
        elif part:
            result.append(int(part))
    return sorted(set(result))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download PhysioNet EEGBCI files via AWS S3 mirror.")
    parser.add_argument(
        "-s",
        "--subjects",
        default="1",
        help="Subject number(s) or range (e.g. '1', '1-5', 'all'). Default: '1'.",
    )
    parser.add_argument(
        "-r",
        "--runs",
        default="1-14",
        help="Run numbers or range (e.g. '1-14', '4,8,12'). Default: '1-14'.",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent download threads. Default: 8.",
    )
    args = parser.parse_args(argv)

    if args.subjects.lower() == "all":
        subjects = list(range(1, N_SUBJECTS + 1))
    else:
        subjects = parse_range(args.subjects)

    runs = parse_range(args.runs)

    print(f"Downloading {len(subjects)} subject(s) × {len(runs)} run(s) using {args.workers} threads...")
    stats = download_data(subjects=subjects, runs=runs, max_workers=args.workers)
    print(f"Complete: {stats['downloaded']} downloaded, {stats['cached']} already cached (total: {stats['total']}).")


if __name__ == "__main__":
    main()
