"""Mirror every asset of this repo's `main` release into a private Hugging Face repo.

The release is the source of truth: new or replaced assets are copied, and files
no longer in the release are removed. Progress is committed batch by batch, so an
interrupted run resumes where it stopped.

Environment:
    HF_TOKEN       Hugging Face token with write access (not needed for DRY_RUN).
    HF_REPO        Target repo id.
    GITHUB_TOKEN   Optional; raises the GitHub API rate limit.
    DRY_RUN=1      Print the plan and change nothing.
    FORCE_DELETES=1  Allow a run to remove more than MAX_DELETE_FRACTION of the files.
"""

import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
from huggingface_hub.errors import EntryNotFoundError
from huggingface_hub.hf_api import RepoFile

GITHUB_REPO = "NevermindNilas/TAS-Models-Host"
RELEASE_TAG = "main"
HF_REPO = os.environ.get("HF_REPO", "NevermindNilas/tas-models-host")
MANIFEST = ".sync-manifest.json"
KEEP_ON_HF = {".gitattributes", "README.md", MANIFEST}

BATCH_BYTES = 6_000_000_000  # Keeps a batch well inside a hosted runner's free disk.
MIN_EXPECTED_ASSETS = 100  # Fewer means the listing is wrong; never mirror that.
MAX_DELETE_FRACTION = 0.10
MAX_TOTAL_BYTES = 95_000_000_000  # Hugging Face's free private quota is 100 GB.
RETRIES = 4


def githubJson(url):
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "tas-mirror"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def listAssets():
    base = f"https://api.github.com/repos/{GITHUB_REPO}/releases"
    release = githubJson(f"{base}/tags/{RELEASE_TAG}")
    assets = {}
    page = 1
    while True:
        batch = githubJson(f"{base}/{release['id']}/assets?per_page=100&page={page}")
        if not batch:
            return assets
        for asset in batch:
            digest = asset.get("digest") or ""
            assets[asset["name"]] = {
                "size": asset["size"],
                "updated_at": asset["updated_at"],
                "sha256": digest.removeprefix("sha256:") or None,
                "url": asset["browser_download_url"],
            }
        page += 1


def fetch(name, asset, destination):
    """Download one asset, returning its SHA-256 once size and digest check out."""
    for attempt in range(1, RETRIES + 1):
        try:
            request = urllib.request.Request(
                asset["url"], headers={"User-Agent": "tas-mirror"}
            )
            digest = hashlib.sha256()
            with urllib.request.urlopen(request, timeout=120) as response:
                with open(destination, "wb") as output:
                    while chunk := response.read(1 << 20):
                        digest.update(chunk)
                        output.write(chunk)
            size = destination.stat().st_size
            if size != asset["size"]:
                raise OSError(f"got {size} bytes, expected {asset['size']}")
            sha256 = digest.hexdigest()
            if asset["sha256"] and asset["sha256"] != sha256:
                raise OSError(f"digest {sha256} != release digest {asset['sha256']}")
            return sha256
        except (OSError, urllib.error.URLError) as error:
            if attempt == RETRIES:
                raise
            print(f"  retry {attempt} for {name}: {error}", flush=True)
            time.sleep(10 * attempt)


def loadManifest(api):
    try:
        path = api.hf_hub_download(HF_REPO, MANIFEST, repo_type="model", force_download=True)
    except EntryNotFoundError:
        return {}
    return json.loads(Path(path).read_text())


def hubFiles(api):
    entries = api.list_repo_tree(HF_REPO, repo_type="model", recursive=True)
    return {entry.path: entry for entry in entries if isinstance(entry, RepoFile)}


def manifestOperation(manifest):
    payload = json.dumps(manifest, indent=0, sort_keys=True).encode()
    return CommitOperationAdd(path_in_repo=MANIFEST, path_or_fileobj=payload)


def planChanges(assets, manifest, hub):
    """Return the asset names to copy and the hub paths to remove."""
    copy = []
    for name, asset in sorted(assets.items()):
        known = manifest.get(name)
        unchanged = (
            known is not None
            and name in hub
            and known["size"] == asset["size"]
            and known["updated_at"] == asset["updated_at"]
        )
        if not unchanged:
            copy.append(name)
    remove = sorted(path for path in hub if path not in assets and path not in KEEP_ON_HF)
    return copy, remove


def copyBatches(api, names, assets, manifest):
    scratch = Path(tempfile.mkdtemp(prefix="tas-mirror-"))
    operations, batchBytes, done = [], 0, 0

    def commit():
        nonlocal operations, batchBytes
        if not operations:
            return
        operations.append(manifestOperation(manifest))
        count = len(operations) - 1
        api.create_commit(
            HF_REPO,
            operations=operations,
            repo_type="model",
            commit_message=f"Mirror {count} release assets",
        )
        print(f"committed {count} files ({done}/{len(names)})", flush=True)
        for leftover in scratch.iterdir():
            leftover.unlink()
        operations, batchBytes = [], 0

    try:
        for name in names:
            asset = assets[name]
            if batchBytes and batchBytes + asset["size"] > BATCH_BYTES:
                commit()
            local = scratch / name
            sha256 = fetch(name, asset, local)
            operations.append(CommitOperationAdd(path_in_repo=name, path_or_fileobj=local))
            manifest[name] = {
                "size": asset["size"],
                "updated_at": asset["updated_at"],
                "sha256": sha256,
            }
            batchBytes += asset["size"]
            done += 1
        commit()
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def verify(api, assets, manifest):
    hub = hubFiles(api)
    problems, hashed = [], 0
    for name, asset in assets.items():
        entry = hub.get(name)
        if entry is None:
            problems.append(f"missing: {name}")
        elif entry.size != asset["size"]:
            problems.append(f"size differs: {name}")
        elif entry.lfs is not None:
            if entry.lfs.sha256 != manifest[name]["sha256"]:
                problems.append(f"hash differs: {name}")
            else:
                hashed += 1
    extras = [path for path in hub if path not in assets and path not in KEEP_ON_HF]
    problems += [f"not in release: {path}" for path in extras]
    print(f"verified {len(assets)} files, {hashed} by SHA-256, the rest by size", flush=True)
    return problems


def main():
    dryRun = os.environ.get("DRY_RUN") == "1"
    forceDeletes = os.environ.get("FORCE_DELETES") == "1"
    api = HfApi()

    assets = listAssets()
    totalBytes = sum(asset["size"] for asset in assets.values())
    print(f"release: {len(assets)} assets, {totalBytes / 1e9:.2f} GB", flush=True)
    if len(assets) < MIN_EXPECTED_ASSETS:
        sys.exit(f"Only {len(assets)} assets listed; refusing to mirror.")
    if totalBytes > MAX_TOTAL_BYTES:
        sys.exit(f"Release is {totalBytes / 1e9:.1f} GB, too close to the 100 GB quota.")

    if not api.repo_info(HF_REPO, repo_type="model").private:
        sys.exit(f"{HF_REPO} is not private; refusing to mirror.")

    manifest = loadManifest(api)
    hub = hubFiles(api)
    copy, remove = planChanges(assets, manifest, hub)
    replaced = [name for name in copy if name in hub]
    copyBytes = sum(assets[name]["size"] for name in copy)
    print(
        f"plan: copy {len(copy)} ({copyBytes / 1e9:.2f} GB, {len(replaced)} replaced), "
        f"remove {len(remove)}",
        flush=True,
    )
    for path in remove:
        print(f"  remove: {path}", flush=True)

    tracked = max(len(hub) - len(KEEP_ON_HF & set(hub)), 1)
    tooManyDeletes = len(remove) / tracked > MAX_DELETE_FRACTION and not forceDeletes
    if dryRun:
        if tooManyDeletes:
            print("Removals would be refused without FORCE_DELETES=1.")
        return

    copyBatches(api, copy, assets, manifest)

    if tooManyDeletes:
        sys.exit(
            f"Refusing to remove {len(remove)} of {tracked} files. "
            "Re-run the workflow with force_deletes if this is intended."
        )

    if remove:
        for path in remove:
            manifest.pop(path, None)
        operations = [CommitOperationDelete(path_in_repo=path) for path in remove]
        operations.append(manifestOperation(manifest))
        api.create_commit(
            HF_REPO,
            operations=operations,
            repo_type="model",
            commit_message=f"Remove {len(remove)} assets no longer in the release",
        )

    if remove or replaced:
        # Old revisions keep counting against the storage quota until squashed.
        api.super_squash_history(HF_REPO, repo_type="model")
        print("squashed history", flush=True)

    problems = verify(api, assets, manifest)
    if problems:
        print("\n".join(problems))
        sys.exit(f"{len(problems)} files failed verification.")
    print("backup is in sync", flush=True)


if __name__ == "__main__":
    main()
