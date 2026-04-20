#!/usr/bin/env python3
"""Download Hugging Face model files with include/exclude filtering and optional proxy support."""

import argparse
import os
import sys
from pathlib import Path
from fnmatch import fnmatch
from urllib.parse import urlparse, urlunparse

import requests
from huggingface_hub import HfApi, hf_hub_download


def matches_patterns(filename: str, patterns: list[str]) -> bool:
    return any(fnmatch(filename, p) for p in patterns)


def build_proxy_url(proxy: str, proxy_user: str | None, proxy_password: str | None) -> str:
    if not proxy_user:
        return proxy
    parsed = urlparse(proxy)
    password_part = f":{proxy_password}" if proxy_password else ""
    netloc = f"{proxy_user}{password_part}@{parsed.hostname}"
    if parsed.port:
        netloc += f":{parsed.port}"
    return urlunparse(parsed._replace(netloc=netloc))


def configure_proxy(proxy: str, proxy_user: str | None, proxy_password: str | None) -> dict:
    proxy_url = build_proxy_url(proxy, proxy_user, proxy_password)
    os.environ["HTTP_PROXY"] = proxy_url
    os.environ["HTTPS_PROXY"] = proxy_url
    return {"http": proxy_url, "https": proxy_url}


def download_model(
    repo_id: str,
    output_dir: str,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    proxies: dict | None = None,
) -> list[str]:
    api = HfApi()
    if proxies:
        session = requests.Session()
        session.proxies.update(proxies)
        api.session = session

    files = list(api.list_repo_files(repo_id))

    selected, skipped = [], []
    for f in files:
        if include and not matches_patterns(f, include):
            skipped.append(f)
            continue
        if exclude and matches_patterns(f, exclude):
            skipped.append(f)
            continue
        selected.append(f)

    if not selected:
        print("No files match the given filters. Exiting.")
        sys.exit(1)

    print(f"Downloading {len(selected)} file(s) from {repo_id}")
    if skipped:
        print(f"Skipping {len(skipped)} file(s): {skipped}")

    out = Path(output_dir)
    model_dir = out / "model"
    results_dir = out / "results"
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []
    failed = []

    for i, filename in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] {filename}", end=" ", flush=True)
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(model_dir),
            )
            downloaded.append(local_path)
            print("OK")
        except Exception as e:
            failed.append(filename)
            print(f"FAILED: {e}")

    print(f"\nDownloaded {len(downloaded)}/{len(selected)} file(s) to {out.resolve()}:")
    for path in downloaded:
        print(f"  {path}")

    if failed or downloaded:
        _write_manifest(results_dir, downloaded, failed)

    if failed:
        sys.exit(1)

    return downloaded


def _write_manifest(results_dir: Path, downloaded: list[str], failed: list[str]) -> None:
    downloaded_txt = results_dir / "downloaded.txt"
    failed_txt = results_dir / "failed.txt"

    downloaded_txt.write_text("\n".join(downloaded) + "\n" if downloaded else "")
    failed_txt.write_text("\n".join(failed) + "\n" if failed else "")

    print(f"\nManifest written to {results_dir}:")
    print(f"  {downloaded_txt}  ({len(downloaded)} file(s))")
    print(f"  {failed_txt}  ({len(failed)} file(s))")


def main():
    parser = argparse.ArgumentParser(
        description="Download Hugging Face model files with optional filtering and proxy support."
    )
    parser.add_argument("repo_id", help="Model repo ID, e.g. moonshotai/Kimi-K2.5")
    parser.add_argument(
        "-o", "--output-dir", default="./output",
        help="Local directory to save files (default: ./output)",
    )
    parser.add_argument(
        "--include", nargs="+", metavar="PATTERN",
        help="Only download files matching these glob patterns, e.g. '*.safetensors' 'config.json'",
    )
    parser.add_argument(
        "--exclude", nargs="+", metavar="PATTERN",
        help="Skip files matching these glob patterns, e.g. '*.bin' 'original/*'",
    )

    proxy_group = parser.add_argument_group("proxy")
    proxy_group.add_argument(
        "--proxy", default=os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY"),
        metavar="URL",
        help="Proxy URL, e.g. http://proxy.corp.com:8080 (also reads HTTPS_PROXY / HTTP_PROXY env vars)",
    )
    proxy_group.add_argument(
        "--proxy-user", default=os.environ.get("PROXY_USER"),
        metavar="USER",
        help="Proxy username (also reads PROXY_USER env var)",
    )
    proxy_group.add_argument(
        "--proxy-password", default=os.environ.get("PROXY_PASSWORD"),
        metavar="PASS",
        help="Proxy password (also reads PROXY_PASSWORD env var)",
    )

    args = parser.parse_args()

    proxies = None
    if args.proxy:
        proxies = configure_proxy(args.proxy, args.proxy_user, args.proxy_password)
        safe_url = build_proxy_url(args.proxy, args.proxy_user, None)
        print(f"Using proxy: {safe_url}")

    download_model(
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        include=args.include,
        exclude=args.exclude,
        proxies=proxies,
    )


if __name__ == "__main__":
    main()
