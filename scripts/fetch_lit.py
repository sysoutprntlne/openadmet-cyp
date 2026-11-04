"""Pull the open-code leaderboard submissions' reports/repos into results/lit/.

Downloads the live regression leaderboard from the openadmet/cyp-challenge
Gradio Space (or reads local CSVs given on the command line), keeps rows with
``Open Code == Yes``, and for each submission:
  * GitHub links -> shallow clone of the repo into ``results/lit/<user>/repo``
    (pinned to the commit when the link carries a 40-char sha), plus the linked
    file path recorded in ``source.txt``.
  * Anything else (blog, hosted page) -> the page saved as ``page.html``.
A ``results/lit/index.csv`` summarizes what was fetched.

Usage:
    python scripts/fetch_lit.py                    # live leaderboard
    python scripts/fetch_lit.py --board interim    # interim leaderboard
    python scripts/fetch_lit.py a.csv b.csv        # local CSVs (newest last)
    python scripts/fetch_lit.py --no-large         # skip repo files over 5 MB (--max-mb N)
"""

import argparse
import csv
import io
import json
import re
import subprocess
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "lit"
SHA = re.compile(r"^[0-9a-f]{40}$")
SPACE_API = "https://openadmet-cyp-challenge.hf.space/gradio_api"
# Gradio endpoints behind the "Download CSV" button on each Overall regression tab
BOARDS = {"live": "partial", "interim": "partial_18"}


def fetch_leaderboard(board):
    """Download the regression leaderboard CSV from the Space; return its text."""
    ep = BOARDS[board]
    req = urllib.request.Request(
        f"{SPACE_API}/call/{ep}",
        data=json.dumps({"data": []}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        event_id = json.load(r)["event_id"]
    with urllib.request.urlopen(f"{SPACE_API}/call/{ep}/{event_id}", timeout=120) as r:
        stream = r.read().decode()
    event, data = None, None
    for line in stream.splitlines():
        if line.startswith("event:"):
            event = line.split(":", 1)[1].strip()
        elif line.startswith("data:") and event == "complete":
            data = json.loads(line.split(":", 1)[1])
    if data is None:
        raise RuntimeError(f"no 'complete' event from {ep}: {stream[:200]}")
    with urllib.request.urlopen(data[0]["url"], timeout=60) as r:
        return r.read().decode()


def open_code_links(texts):
    """Return {username: link}; later leaderboards (listed last) override earlier ones."""
    links = {}
    for text in texts:
        for row in csv.DictReader(io.StringIO(text)):
            link = row["Model Report Link"].strip()
            if row["Open Code"].strip().lower() == "yes" and link.startswith("http"):
                links[row["Username"]] = link
    return links


def slug(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def git(repo_dir, *args, **kw):
    return subprocess.run(["git", *args], cwd=repo_dir, check=True, **kw)


def checkout_small(repo_dir, max_bytes):
    """Check out only files under max_bytes; list the skipped ones in LARGE_FILES_SKIPPED.txt."""
    tree = git(repo_dir, "ls-tree", "-r", "-l", "-z", "HEAD", capture_output=True, text=True).stdout
    keep, skipped = [], []
    for entry in filter(None, tree.split("\0")):
        meta, path = entry.split("\t", 1)
        size = meta.split()[3]
        if size != "-" and int(size) > max_bytes:
            skipped.append(f"{int(size) / 1e6:.1f}M\t{path}")
        else:
            keep.append(path)
    git(repo_dir, "checkout", "-q", "HEAD", "--pathspec-from-file=-", "--pathspec-file-nul",
        input="\0".join(keep), text=True)
    if skipped:
        (repo_dir.parent / "LARGE_FILES_SKIPPED.txt").write_text("\n".join(skipped) + "\n")


def clone_github(link, dest, max_bytes=None):
    parts = urlparse(link).path.strip("/").split("/")
    owner, repo = parts[:2]
    ref = parts[3] if len(parts) > 3 and parts[2] in ("blob", "tree") else None
    sub = "/".join(parts[4:]) if ref else ""
    repo_dir = dest / "repo"
    url = f"https://github.com/{owner}/{repo}.git"
    if repo_dir.exists():
        return f"{owner}/{repo} (already cloned)", sub
    # With a size cap, use a blobless-for-big-files partial clone and check out only small files.
    filt = ["--filter", f"blob:limit={max_bytes + 1}", "--no-checkout"] if max_bytes else []
    if ref and SHA.match(ref):
        repo_dir.mkdir(parents=True)
        git(repo_dir, "init", "-q")
        git(repo_dir, "remote", "add", "origin", url)
        git(repo_dir, "fetch", "-q", "--depth", "1", *filt[:2], "origin", ref)
        git(repo_dir, "update-ref", "HEAD", "FETCH_HEAD") if max_bytes else git(repo_dir, "checkout", "-q", "FETCH_HEAD")
    else:
        subprocess.run(["git", "clone", "-q", "--depth", "1", *filt, url, str(repo_dir)], check=True)
    if max_bytes:
        checkout_small(repo_dir, max_bytes)
    return f"{owner}/{repo}", sub


def save_page(link, dest):
    req = urllib.request.Request(link, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        (dest / "page.html").write_bytes(r.read())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csvs", nargs="*", type=Path, help="local leaderboard CSVs (newest last); default: fetch live")
    ap.add_argument("--board", choices=sorted(BOARDS), default="live", help="which board to download when no CSVs given")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--no-large", action="store_true", help="skip files over --max-mb when cloning repos")
    ap.add_argument("--max-mb", type=float, default=5, help="size cap for --no-large (default 5)")
    args = ap.parse_args()

    if args.csvs:
        texts = [p.read_text() for p in args.csvs]
    else:
        texts = [fetch_leaderboard(args.board)]
    links = open_code_links(texts)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "leaderboard.csv").write_text(texts[-1])
    index = []
    for user, link in links.items():
        dest = args.out / slug(user)
        dest.mkdir(exist_ok=True)
        (dest / "source.txt").write_text(link + "\n")
        try:
            if urlparse(link).netloc == "github.com":
                what, sub = clone_github(link, dest, int(args.max_mb * 1e6) if args.no_large else None)
            else:
                save_page(link, dest)
                what, sub = "page.html", ""
            status = "ok"
        except Exception as e:  # keep going; report at the end
            what, sub, status = "", "", f"failed: {e}"
        print(f"{status:>8}  {user}  {link}")
        index.append({"user": user, "link": link, "fetched": what, "path": sub, "status": status})

    with open(args.out / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["user", "link", "fetched", "path", "status"])
        w.writeheader()
        w.writerows(index)
    failed = [r["user"] for r in index if r["status"] != "ok"]
    print(f"\n{len(index) - len(failed)}/{len(index)} fetched into {args.out}")
    if failed:
        print("failed:", ", ".join(failed))


if __name__ == "__main__":
    main()
