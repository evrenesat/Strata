#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/core/conversation_kv_layout.hpp"
#include "strata/core/kv_transfer.hpp"
#include "strata/core/conversation_file.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/session.hpp"

#include <functional>
#include <string>

namespace strata::core {

// Caller synchronizes the device before saving, and after restoring all layers.
// include_index is false for the draft layer (its attention has no indexer).
size_t conversation_kv_bytes(const QsaState& state, const ModelGeometry& g, int64_t upto, bool include_index);
// A nonzero unchanged_tokens is valid only for storage retained from an image
// actually restored into this session, bounded by every subsequent rewrite.
// Equal token IDs alone do not establish that its K/V bytes are unchanged.
bool conversation_kv_save(ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                          int64_t upto, bool include_index, std::string& error,
                          int64_t unchanged_tokens = 0, size_t* reused_bytes = nullptr);
bool conversation_kv_capture_bytes(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                                   int64_t upto, bool include_index, size_t& bytes, std::string& error);
// No CUDA calls or destination writes. Used for whole-session prevalidation.
bool conversation_kv_validate(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                              int64_t upto, bool include_index, std::string& error);
bool conversation_kv_restore(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                             int64_t upto, bool include_index, std::string& error);
// Diagnostic read-back after a synchronized restore. Uses 64 KiB of stack
// workspace, compares authoritative bytes and resident draft-ring pages, and
// fingerprints the authoritative payload only. Never changes model state.
bool conversation_kv_verify(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                            int64_t upto, bool include_index, uint64_t& fingerprint, std::string& error,
                            bool restored_residency = true);
// Disk save without a host copy: the layer's authoritative K/V as a streamed source (same bytes and metadata as
// conversation_kv_save).  Caller synchronizes the device first and keeps the state untouched while it is read.
bool conversation_kv_source(SessionKvSource& source, const QsaState& state, const ModelGeometry& g,
                            int64_t upto, bool include_index, std::string& error);
// How a restore ended: `invalid` is a refusal before any write, `transfer_failed` may already have written part
// of the destination, and the caller must not continue inference from such a session.
enum class ConversationRestore { restored, invalid, transfer_failed };

// --- the disk path's halves: visit a payload as verified blocks, write one into the live pools ----
// Pushes a RAM image's own segmented buffers as complete blocks, in part order, with no staging copy.  A verified
// spill read pushes its blocks through the same sink instead.
bool conversation_kv_visit(const ConversationKv& image, const KvBlockSink& sink, std::string& error);
// The live destination for a restore: one verified block is copied into the authoritative pool for its part, bounds
// checked against the layout.  The caller synchronizes the device before and after, exactly as the RAM restore
// requires.  An empty function (with `error` set) means the state cannot be a destination.
KvBlockSink conversation_kv_writer(const QsaState& state, const ConversationKvLayout& layout, std::string& error);
// After every part was written: invalidate the streaming map and refill the resident ring from the restored
// authoritative pools, exactly as the RAM restore does.
bool conversation_kv_finish_restore(const QsaState& state, const ModelGeometry& g, int64_t upto, std::string& error);

// What a transfer restore ended as, in both vocabularies: the RAM-shaped outcome for existing callers and the
// transaction detail (refused / lost / fatal) that decides between a typed loss and a poisoned engine.
struct ConversationKvRestoreResult {
    ConversationRestore outcome = ConversationRestore::invalid;
    KvTransferOutcome detail = KvTransferOutcome::refused;
    uint64_t verified_bytes = 0;
    std::string error;
};
// Restore any K/V source through the shared transaction rules: verify, then write, then count, under one mutation
// boundary, followed by the residency finish.  `invalid` means nothing was written.
ConversationKvRestoreResult conversation_kv_restore_source(const QsaState& state, const ModelGeometry& g, int64_t upto,
                                                           bool include_index, const KvDriveFn& drive);

// One layer's destination for a multi-layer restore.  Main layers come first in the vector, the draft entry (if any)
// last, which is the order a capsule's manifest uses.
struct ConversationKvTarget {
    const QsaState* state = nullptr;
    bool draft = false;
    int64_t upto = 0;   // this layer's extent: the main prefix for a QSA layer, the speculative extent for the draft
};
// Restore every layer of a capsule from ONE verified pass under ONE mutation boundary: all layouts are validated and
// every destination prepared before the first write, each block goes to its own layer, and the residency finish runs
// for every layer after the last write.  `invalid` means nothing was written.
// `before_draft`, when set, runs exactly once immediately before the draft entry's first block is written - after
// every main block has been.  That is where a capsule restores the main layers' RECURRENT state, so the order stays
// K/V, then recurrent state, then the draft's own K/V.  It is not called when the capsule has no draft entry.
ConversationKvRestoreResult conversation_kv_restore_layers(const ModelGeometry& g,
                                                           const std::vector<ConversationKvTarget>& targets,
                                                           const KvLayerDriveFn& drive,
                                                           const std::function<bool(std::string&)>& before_draft = {});

// The bytes of each K/V part (k, v, k_scale, v_scale, pooled) a snapshot of `upto` tokens holds: no state read.
bool conversation_kv_part_sizes(const QsaState& state, const ModelGeometry& g, int64_t upto, bool include_index,
                                std::array<uint64_t, 5>& sizes, std::string& error);
// Disk sessions: the read limits this session can ever restore - its geometry and layer range, at most
// min(max_tokens, its cells) tokens, `max_checkpoints` checkpoints, the exact running-state sizes and the K/V part
// sizes at that many tokens - so session_file_read refuses an oversized or foreign file before it allocates.
bool conversation_session_read_limits(SessionReadLimits& limits, const SessionState& session, const ModelGeometry& g,
                                      const QsaState& draft, uint64_t max_tokens, uint64_t max_checkpoints,
                                      std::string& error);

struct ConversationStateSizes {
    size_t gdn = 0, ple = 0, tail = 0, dead = 0, block_pos = 0;
};
/// Whole-model sizes: `gdn` covers every GDN layer, the indexer sizes are per QSA layer.
bool conversation_state_sizes(const ModelGeometry& g, ConversationStateSizes& sizes, std::string& error);
/// The same for one session's layer carve (#216): `gdn` covers its `gdn_alloc` rows; the per-QSA-layer sizes
/// apply to each owned state [qsa_ord0, qsa_ord0 + qsa_alloc).  Rejects an inconsistent carve.  Checkpoints,
/// snapshots and their validation all use this: a session saves and restores only the state it owns.
bool conversation_session_sizes(const ModelGeometry& g, const SessionState& session, ConversationStateSizes& sizes,
                                std::string& error);
bool conversation_checkpoint_validate(const ConversationCheckpoint& checkpoint, const SessionState& session,
                                      const ModelGeometry& g, std::string& error);
bool conversation_checkpoint_save(ConversationCheckpoint& checkpoint, const SessionState& session,
                                  const ModelGeometry& g, std::string& error);
bool conversation_checkpoint_restore(const ConversationCheckpoint& checkpoint, SessionState& session,
                                     const ModelGeometry& g, std::string& error);
// Default-off native diagnostics call these only at a drained ownership boundary.
// Streaming readback uses bounded64KiB scratch, no second full image.
bool conversation_snapshot_verify(const SavedConversation&,const SessionState&,const ModelGeometry&,
                                  uint64_t& fingerprint,std::string& error);
bool conversation_live_fingerprint(const SessionState&,const ModelGeometry&,int64_t upto,
                                   uint64_t& fingerprint,std::string& error);

struct ConversationView {
    const std::vector<int32_t>& ids;
    const std::vector<ConversationImageKey>& images;
    const std::vector<ConversationCheckpoint>& checkpoints;
    bool cvec;
};
bool conversation_snapshot_bytes(const ConversationView& view, const SessionState& session,
                                 const ModelGeometry& g, const QsaState& draft, size_t& bytes, std::string& error);
bool conversation_snapshot_capture_bytes(const ConversationKvReuse& reuse, const ConversationView& view,
                                         const SessionState& session, const ModelGeometry& g,
                                         const QsaState& draft, size_t& bytes, std::string& error);
// The capture estimate includes retained capacity and transient segment directories;
// only estimate - reuse.bytes() requires additional physical RAM. Capture consumes
// the uniquely owned reusable buffers, including on failure.
// Caller admits the estimate before invoking capture. Allocation failures propagate
// to the RAM policy; the active session is never modified by capture.
bool conversation_snapshot_save(SavedConversation& image, const ConversationView& view,
                                const SessionState& session, const ModelGeometry& g,
                                const QsaState& draft, std::string& error,
                                ConversationKvReuse reuse = {}, size_t* reused_bytes = nullptr);
// Disk save without capturing the K/V on the host: `meta` gets everything but the K/V (running state copied,
// checkpoints as given by the view), `sources` one streamed source per QSA layer then the draft.  Caller has
// synchronized and must not run the session until the file is written.
bool conversation_snapshot_sources(SavedConversation& meta, std::vector<SessionKvSource>& sources,
                                   const ConversationView& view, const SessionState& session,
                                   const ModelGeometry& g, const QsaState& draft, std::string& error);
// The same without the draft layer's K/V (`draft == nullptr`): `sources` holds the session's own QSA layers only,
// matching a snapshot validated and restored with the nullable-draft calls below.  A layer split's later stages park
// this way.
bool conversation_snapshot_sources(SavedConversation& meta, std::vector<SessionKvSource>& sources,
                                   const ConversationView& view, const SessionState& session,
                                   const ModelGeometry& g, const QsaState* draft, std::string& error);
bool conversation_snapshot_validate(const SavedConversation& image, const SessionState& session,
                                    const ModelGeometry& g, const QsaState& draft, std::string& error);
// The disk capsule's two halves.  Validation: the same metadata checks, with the image required to carry NO resident
// K/V (it lives in the capsule file, and the caller proves those layers against the session's targets before any
// write).  Restore: the running state and checkpoints, with no K/V call at all - the caller restores the K/V from
// the capsule first, so the layer-by-layer order stays main K/V, main recurrent state, MTP K/V, MTP resident state.
bool conversation_snapshot_validate_resident(const SavedConversation& image, const SessionState& session,
                                             const ModelGeometry& g, std::string& error);
bool conversation_snapshot_restore_resident(const SavedConversation& image, SessionState& session,
                                            const ModelGeometry& g, std::string& error);
// Invalid images are rejected before any CUDA call/write. Transfer failure may
// leave partial state: caller MUST NOT continue inference from that session.
ConversationRestore conversation_snapshot_restore(const SavedConversation& image, SessionState& session,
                                                   const ModelGeometry& g, const QsaState& draft,
                                                   std::string& error);

// The same with `draft == nullptr`: an image WITHOUT the draft layer's K/V (kv holds the session's own QSA layers
// only).  A layer split's later stages park this way; the draft ring is saved once, with the first stage's image.
// An image is validated and restored with the same kind of call it was saved with (a K/V layer count mismatch is
// rejected as invalid).
bool conversation_snapshot_bytes(const ConversationView& view, const SessionState& session, const ModelGeometry& g,
                                 const QsaState* draft, size_t& bytes, std::string& error);
bool conversation_snapshot_capture_bytes(const ConversationKvReuse& reuse, const ConversationView& view,
                                         const SessionState& session, const ModelGeometry& g, const QsaState* draft,
                                         size_t& bytes, std::string& error);
bool conversation_snapshot_save(SavedConversation& image, const ConversationView& view, const SessionState& session,
                                const ModelGeometry& g, const QsaState* draft, std::string& error,
                                ConversationKvReuse reuse = {}, size_t* reused_bytes = nullptr);
bool conversation_snapshot_validate(const SavedConversation& image, const SessionState& session,
                                    const ModelGeometry& g, const QsaState* draft, std::string& error);
ConversationRestore conversation_snapshot_restore(const SavedConversation& image, SessionState& session,
                                                   const ModelGeometry& g, const QsaState* draft, std::string& error);

} // namespace strata::core
