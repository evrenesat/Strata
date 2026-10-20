# Verification procedure for the priority parking feature

This is the repeatable procedure behind the results quoted in PR #1. Raw
evidence receipts (hashes, thermal logs, guard records) live in a private
operations repository; everything needed to *reproduce* the checks is in this
repository plus one GPU host.

## Builds

- CPU ownership/validation tests (no GPU needed):
  `cmake -S . -B build-cpu -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_CONVERSATION_TESTS=ON`, then
  `ctest --test-dir build-cpu --output-on-failure -R 'generation_capsule_test|conversation_memory_test|conversation_validation_test'`.
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
| `generation_capsule_test`, `conversation_*_test` (C++) | capsule ownership/lifetimes, memory admission incl. cgroup v2/v1 bounds, snapshot validation refusals |

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
