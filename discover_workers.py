from huggingface_hub import HfApi

REPO = "builddotai/Egocentric-100K"
api = HfApi()

# Choice: list full repo tree filtered by factory prefixes.
# Alternative was downloading the repo file list as JSON, but the API
# handles pagination automatically.
factories = [f"factory{i:03d}" for i in range(1, 12)]

for factory in factories:
    print(f"\n{'='*60}")
    print(factory)
    print(f"{'='*60}")

    # List all files under this factory prefix
    files = list(api.list_repo_tree(
        repo_id=REPO,
        repo_type="dataset",
        path_in_repo=factory,
    ))

    # Find worker directories
    workers = {}
    for f in files:
        # f.rfilename gives the path like "factory001/worker002"
        name = f.path if hasattr(f, 'path') else str(f)
        parts = name.split("/")
        if len(parts) >= 2 and parts[1].startswith("worker"):
            worker = parts[1]
            if worker not in workers:
                workers[worker] = []

    # For each worker, find tar files
    for worker in sorted(workers):
        worker_path = f"{factory}/{worker}"
        worker_files = list(api.list_repo_tree(
            repo_id=REPO,
            repo_type="dataset",
            path_in_repo=worker_path,
        ))
        tars = [
            f.path if hasattr(f, 'path') else str(f)
            for f in worker_files
            if str(f.path if hasattr(f, 'path') else f).endswith(".tar")
        ]
        print(f"  {worker}: {len(tars)} tar(s)")
        for t in sorted(tars):
            print(f"    {t}")