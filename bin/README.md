# bin/ — host provisioning and the retired GPU job queue

*Updated: 2026-10-26 UTC*

**Status: legacy.** The Hopper-polling GPU job queue (`gpu_daemon.sh`, `gpu_runner.sh`,
`queue_jobs.sh`, `gpu_queue.sh`) and the host provisioning scripts
(`provision_gpu_host.sh`, `setup_gpu_host.sh`, `setup_hf_cache.sh`) were retired for new
work in 2026 — see "GPU Compute — Forward Policy" in `Rosetta_Program/CLAUDE.md`. They are
kept here unchanged so that completed runs recorded in Hopper and on Hugging Face remain
reproducible and so the daemon can still be maintained for legacy outputs.

Do not build new science on them. Do not submit new `gpu-job` tasks.

| Script | Purpose | Notes |
|---|---|---|
| `gpu_daemon.sh` | Resident daemon: polls Hopper for `gpu-job` tasks, runs them FIFO | bash 3.2 / macOS compatible; uses public `origin` URLs only |
| `gpu_runner.sh` | Per-job runner invoked by the daemon | |
| `queue_jobs.sh` | Load a jobs file (`# ---` = dependency barrier) into Hopper | |
| `gpu_queue.sh` | Show the queue in run order | |
| `provision_gpu_host.sh` | One-command at-will provisioning of a fresh GPU host | |
| `setup_gpu_host.sh` | Interactive host bootstrap (venv, clones, HF cache) | prints a `~/Source/Rosetta_Program/...` hint that only applies on the dev machine |
| `setup_hf_cache.sh` | Persistent `HF_HOME` on a coder-style host (`/home/coder`) | host-specific |
| `sync_models.sh` | Host-to-host model sync | |
| `sync_results.sh` | Pull `rosetta_data/` results from hosts in `sync_hosts.conf` | `DEST` is the dev-machine path; Hugging Face is now the data system of record |

None of these scripts are part of the Python package or its tests. `pip install rosetta_tools`
does not ship them.
