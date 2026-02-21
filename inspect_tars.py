import tarfile
from collections import defaultdict
from huggingface_hub import hf_hub_download

REPO = "builddotai/Egocentric-100K"

TARS = {
    "factory001": "factory001/worker001/part000.tar",
    "factory002": "factory002/worker001/part152.tar",
    "factory003": "factory003/worker001/part187.tar",
    "factory004": "factory004/worker001/part200.tar",
    "factory005": "factory005/worker001/part278.tar",
    "factory006": "factory006/worker001/part345.tar",
    "factory007": "factory007/worker001/part362.tar",
    "factory008": "factory008/worker001/part449.tar",
    "factory009": "factory009/worker001/part566.tar",
    "factory010": "factory010/worker001/part682.tar",
    "factory011": "factory011/worker001/part728.tar",
}


def inspect_tar(tar_path):
    workers = defaultdict(lambda: {"mp4": [], "json": []})

    with tarfile.open(tar_path) as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            name = m.name
            # Choice: parse worker from filename (factory_XXX_worker_YYY_ZZZZ.ext)
            # Alternative was parsing directory structure, but filenames are more reliable
            base = name.split("/")[-1]
            parts = base.split("_")
            if len(parts) >= 4:
                worker = f"{parts[2]}_{parts[3]}"  # worker_001
                if base.endswith(".mp4"):
                    workers[worker]["mp4"].append(base)
                elif base.endswith(".json"):
                    workers[worker]["json"].append(base)

    return workers


def main():
    # Step 1: download all tars (cached by huggingface_hub)
    tar_paths = {}
    for factory, tar_name in TARS.items():
        print(f"Downloading {factory} ({tar_name})...")
        tar_paths[factory] = hf_hub_download(
            repo_id=REPO,
            filename=tar_name,
            repo_type="dataset",
        )
    print()

    # Step 2: inspect each tar
    for factory in sorted(tar_paths):
        print(f"{'='*60}")
        print(f"{factory} — {TARS[factory]}")
        print(f"{'='*60}")

        workers = inspect_tar(tar_paths[factory])
        for worker in sorted(workers):
            mp4s = sorted(workers[worker]["mp4"])
            jsons = sorted(workers[worker]["json"])
            print(f"  {worker}: {len(mp4s)} mp4s, {len(jsons)} jsons")
            print(f"    first: {mp4s[0]}")
            print(f"    last:  {mp4s[-1]}")
        print()


main()