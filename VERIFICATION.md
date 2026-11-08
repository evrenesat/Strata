# Verification procedure for the priority parking feature

This is the repeatable procedure behind the results quoted in PR #1. Raw
evidence receipts (hashes, thermal logs, guard records) live in a private
operations repository; everything needed to *reproduce* the checks is in this
repository plus one GPU host.

## Builds

- CPU ownership/validation tests (no GPU needed):
  `cmake -S . -B build-cpu -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_CONVERSATION_TESTS=ON`, then
  `ctest --test-dir build-cpu --output-on-failure -R 'generation_capsule_test|conversation_memory_test'`.
- HIP engine (tested on gfx1100 / RX 7900 XTX, ROCm 7.14): configure with
  `STRATA_ENABLE_HIP=ON`, `CMAKE_HIP_ARCHITECTURES=gfx1100`,
  `STRATA_BUILD_CONVERSATION_TESTS=ON`; build the `strata` target and the
  conversation test targets.

## In-repo test coverage

| Suite | Covers |
|---|---|
| `serve/test_priority_scheduler.py` | lease renewal/expiry races, queue bounds, queued-cancellation wake-up, persistent-resume-stranding policy, lease collapse after failed admission, backfill gating, parked cancellation, engine death |
| `serve/test_priority_protocol.py` | PGEN/PPAUSE/PRESUME/PCANCEL framing, PAUSED/RESUMED/PREFUSED handling, BLOCK-before-ACK ordering |
| `serve/test_priority_stream.py` | SSE keep-alive/status comments, parser continuity across a pause |
| `serve/test_priority_auth.py` | role file validation, constant-time matching, endpoint allowlist, malformed entries fail startup |
| `generation_capsule_test`, `conversation_*_test` (C++) | capsule ownership/lifetimes, memory admission incl. cgroup v2/v1 bounds (and the reclaimable-file-cache credit), snapshot validation refusals |
| `kv_spill_test` (C++, host-only) | anonymous capsule storage: pinned directory checks, manifest arithmetic and revalidation, streaming writer with per-block digests and writeback windows, the two-buffer verified reader, injected I/O faults, move/close lifetimes |

## GPU scenario matrix (the real evidence)

All scenarios compare an interrupted run against two uninterrupted
same-profile baselines, requiring equal emitted tokens, parser output,
SSE event stream, usage and finish reason; saved-boundary state readback
(main/draft KV, GDN/PLE, indexer, MTP lookahead, sampler/driver/checkpoint
accounting) must match exactly. Terminal speculative tails beyond EOS are
reported separately (they vary even between uninterrupted runs).

1. **4K exactness matrix**: greedy and sampled (temperature 1, top_p 0.95,
   top_k 20, fixed seed) pause at mid-prefill and at decode >=1; unfinished
   tool-call JSON; reasoning/UTF-8 boundaries; EOS and output-cap edges;
   active penalty history; memory refusal; parked cancellation; fatal
   partial restore. Plus one coupled-sampler profile with nonempty q/history.
2. **Default-suffix HTTP continuity (4K)**: two baselines + prefill and
   decode interruptions through the real HTTP/SSE path with original
   response/socket/parser ownership; production (adaptive cache) profile
   with checkpoint restore and pending-copy drain.
3. **32K**: exact and production profiles, prefill16384/full32768 and decode
   interruptions, three strict transfers; production restores 2 checkpoints
   and drains 96 pending adaptive copies.
4. **245,760-token input (~94% of a 262144 context)**: two uninterrupted
   baselines (identical token/parser hashes) + one full park→VIP→resume
   cycle with byte-exact state restore and clean resumed decode. Known
   limit: a *second consecutive* park of the same near-max request can
   exceed RAM admission (~3.9 GB capsule + 2.5 GB floor on a 64 GB host)
   and fails closed with `PREFUSED`.
5. **Live two-key separation**: a long background generation with the
   ordinary key; mid-generation VIP request (role key) → first visible
   answer in 1.43 s, complete in 1.67 s; identical request with the
   ordinary key queues behind the generation (313 s FIFO control);
   background resumed with a 0.03 s chunk gap, one POST, zero replay;
   VIP key rejected off its three-endpoint allowlist.
6. **Real client prefix** (66 KB system/tools prefix): warm first visible
   answer 12.1 s; ~53 s cold after an engine restart (intrinsic prefill
   cost, not scheduling).

Acceptance for any rerun: no skipped required cases, strict equality on the
above fields, honest failure retention (a `PREFUSED`/timeout is a failure,
never a retry-until-pass), original service restored after isolated runs.

## Still open

- Preemption latency distribution at 4K/32K/128K/near-max contexts and
  under repeated parking (single data point at 121 KB prompt so far).
- Native pause with pending UTF-8 bytes was never observed; covered by
  deterministic tokenizer/parser-identity tests only.

## Parked-capsule spill (opt-in, server side)

The parked capsule holds the suspended request's K/V.  Its disk fallback writes that K/V to ONE anonymous temporary
file when RAM admission refuses it, so the queued VIP can run instead of the background request being lost, and reads
it back verified before any live state is touched.  The RAM capsule stays the default path.

Configuration lives in the config's `priority` section and is validated by the server before anything is spawned:

| Key | Meaning |
|---|---|
| `spill_dir` | absolute directory for the capsule file.  Must exist (or be creatable as the final missing component), be owned by this process, have mode `0700`, and sit on ext4, XFS or Btrfs |
| `spill_mode` | `off` (default without a directory), `fallback` (default with one: try RAM first), `always` (**testing only**: skip the RAM attempt but keep every real memory, disk and time check) |
| `spill_min_free_mib` | the extra free space required beyond the payload, in MiB (default 4096; `0` removes only the extra reserve, never the payload requirement) |

Spill is never enabled by installing a capable binary: it needs both a directory and a serving mode other than `off`.
A spill-enabled start requires the engine to advertise `priority_spill=1` after READY on **every** start, including a
restart or a lazy reload; an engine without it is refused (`priority spill requires an engine that advertises
priority_spill=1`).  `always` prints a testing-only warning at startup.

This repository's server implements and tests the configuration, the capability handshake, the `PAUSED ... spill=1`
capability token and the typed loss (`ERR priority_spill_lost` becomes a `priority_spill_lost` stream error on Chat
Completions, Responses and Anthropic).  **The engine-side capsule spill is not in this tree yet**: a spill-enabled
start is therefore refused against the current engine, which is the correct fail-closed behaviour rather than a silent
half-feature.

### Containers

The capsule file is an `O_TMPFILE` inode, so it has no name, nothing to sweep and nothing left behind after a crash -
but it does need a real Linux filesystem under it.  A container must bind-mount a directory from ext4, XFS or Btrfs
storage whose owner and mode (0700) are correct **as seen inside the container**; tmpfs, ramfs, network and FUSE
filesystems are refused at startup, and the container's own overlayfs writable layer is not supported.  Naming a
Docker volume does not prove its backing filesystem - check the mounted directory from inside the container.

### Policies worth knowing

- **Capture has a publish budget of 60 seconds** from the moment the pause is accepted; a capsule whose estimated
  write would exceed it is refused *before* any disk write and the background keeps running.  A blocked syscall can
  still exceed the deadline, and the engine's existing watchdog covers that case.
- **Writes are buffered and written back every 64 MiB** (`sync_file_range` then `posix_fadvise`), and the final tail
  is flushed before the capsule is published, so captured bytes do not pile up as dirty page cache.
- **The capture-time estimate is deliberately conservative**: it assumes at most 128 MiB/s even when the disk measured
  faster, so a fast disk can never make a prediction optimistic.  With the 60 s budget that bounds a spillable payload
  at roughly 4.7 GiB - comfortably above a near-limit 262K-token capsule (~4 GB), which is what the feature is for -
  and the pause drain and a refused RAM attempt both eat into the same budget.
- **The read cache is dropped as it goes**: each restored block's pages are released after the block is consumed, so a
  long restore does not leave the whole payload in the page cache behind work that is itself under memory pressure.
- **The write counter is logical bytes**, including partial writes of attempts that were later refused.  It is not a
  measure of physical device writes, and it is not framed as drive wear.
- **Corruption handling is one-pass**: every block is verified against its digest before it is used, and any recovery
  decision happens before the first live write.  A fault discovered *after* live state was written is fatal rather
  than recoverable, because there is no safe way to un-restore a half-written engine.  These digests detect
  accidental corruption; they are not a tamper check and closing the file is not secure erasure.
- **Measured memory, not assumptions**: capture is expected to add one 16 MiB staging block, restore two 16 MiB
  buffers plus a small reader allowance, and each writeback window at most 64 MiB.  Verification measures anonymous,
  dirty and writeback deltas per phase against those bounds instead of asserting throughput from the design.

Verification status: the host-side pieces above are covered by the CPU suites in this file.  The GPU/operational
matrix for actual spills (a 245,760-token request with five forced park/restore cycles, a real capacity refusal, the
fault cases and cleanup) requires the engine side and is **not yet executed**.
