#!/usr/bin/env bash
# gpu_queue.sh — Show the gpu-job queue in FIFO order.
#
# Usage: bash ~/rosetta_tools/bin/gpu_queue.sh
#
# Written: 2026-04-21 UTC

hopper sync 2>/dev/null || true

hopper --json task list --tag gpu-job 2>/dev/null \
    | jq -r '
        sort_by(.created_at)
        | .[]
        | select(.status != "cancelled")
        | [
            .id[0:8],
            (if .status == "in_progress" then "running" else .status end),
            (if .assigned_to then (.assigned_to | split(":") | .[1:] | join(":")) else "-" end),
            .title
          ]
        | @tsv' \
    | awk -F'\t' '{printf "%-10s %-12s %-12s %s\n", $1, $2, $3, $4}'
