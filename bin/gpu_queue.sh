#!/usr/bin/env bash
# gpu_queue.sh — Show the gpu-job queue in FIFO order.
#
# Usage: bash ~/rosetta_tools/bin/gpu_queue.sh
#
# Written: 2026-04-21 UTC

hopper sync 2>/dev/null || true

printf "%-10s %-12s %-12s %-10s %s\n" "ID" "STATUS" "HOST" "NEEDS" "TITLE"
hopper --json task list --tag gpu-job 2>/dev/null \
    | jq -r '
        sort_by(.created_at)
        | .[]
        | select(.status != "cancelled")
        | (((.tags // []) | map(select(test("^gpus[0-9]+$")))
            | if length > 0 then (.[0] | ltrimstr("gpus")) else "1" end)) as $g
        | (((.tags // []) | map(select(startswith("host-")))
            | if length > 0 then ("@" + (.[0] | ltrimstr("host-"))) else "" end)) as $h
        | [
            .id[0:8],
            (if .status == "in_progress" then "running" else .status end),
            (if .assigned_to then (.assigned_to | split(":") | .[1:] | join(":")) else "-" end),
            ($g + "gpu" + (if $h != "" then " " + $h else "" end)),
            .title
          ]
        | @tsv' \
    | awk -F'\t' '{printf "%-10s %-12s %-12s %-10s %s\n", $1, $2, $3, $4, $5}'
