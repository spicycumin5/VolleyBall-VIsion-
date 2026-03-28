/**
 * clusterActions.js
 *
 * Transforms raw per-frame action detections into consolidated clips
 * suitable for a Playlist component.
 *
 * Raw input:  [{ player_id, action, frame, action_conf }, ...]
 *   — Many rows per real-world event (e.g. 15 frames of "block" for one play)
 *
 * Output:     [{ id, title, action, playerId, start, end, peakConf, frameStart, frameEnd }, ...]
 *   — One entry per distinct event, with time range in seconds
 *
 * The algorithm:
 *   1. Filter out low-confidence noise
 *   2. Group by (player_id + action)
 *   3. Sort each group by frame
 *   4. Cluster consecutive frames within a gap tolerance into one event
 *   5. Convert frame ranges → time ranges using fps
 *   6. Sort all events chronologically
 */

/**
 * @param {Array} rawActions   — array of { player_id, action, frame, action_conf }
 * @param {Object} opts
 * @param {number} opts.fps           — video frame rate (default 30)
 * @param {number} opts.minConf       — minimum action_conf to keep (default 0.3)
 * @param {number} opts.gapFrames     — max frame gap to still merge into one event (default 10)
 * @param {number} opts.padSeconds    — seconds of padding around the clip for context (default 1)
 * @param {number} opts.minClipFrames — discard clusters shorter than this (default 2)
 * @returns {Array} clips shaped for Playlist: { id, title, action, playerId, start, end, ... }
 */


export function clusterActions(rawActions, opts = {}) {
  const {
    fps = 30,
    minConf = 0.01,
    gapFrames = 10,
    padSeconds = 1,
    minClipFrames = 1,
  } = opts;
 
  // 1. Filter low-confidence detections
  const filtered = rawActions.filter((d) => d.action_conf >= minConf);
 
  // 2. Group by (player_id, action)
  const groups = new Map();
  for (const d of filtered) {
    const key = `${d.player_id}__${d.action}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(d);
  }
 
  // 3–4. Cluster consecutive frames within each group
  const clusters = [];
 
  for (const [key, detections] of groups) {
    // Sort by frame
    detections.sort((a, b) => a.frame - b.frame);
 
    let clusterStart = 0;
    for (let i = 1; i <= detections.length; i++) {
      const gapExceeded =
        i === detections.length ||
        detections[i].frame - detections[i - 1].frame > gapFrames;
 
      if (gapExceeded) {
        const slice = detections.slice(clusterStart, i);
 
        if (slice.length >= minClipFrames) {
          const peakConf = Math.max(...slice.map((d) => d.action_conf));
          clusters.push({
            playerId: slice[0].player_id,
            action: slice[0].action,
            frameStart: slice[0].frame,
            frameEnd: slice[slice.length - 1].frame,
            peakConf,
            count: slice.length,
          });
        }
 
        clusterStart = i;
      }
    }
  }
 
  // 5–6. Convert to clip format, sort chronologically
  const clips = clusters
    .sort((a, b) => a.frameStart - b.frameStart)
    .map((c, i) => {
      const startSec = Math.max(0, c.frameStart / fps - padSeconds);
      const endSec = c.frameEnd / fps + padSeconds;
      const actionLabel = c.action.charAt(0).toUpperCase() + c.action.slice(1);
 
      return {
        id: i + 1,
        //title: `Player ${c.playerId} — ${actionLabel}`,
        title: `${actionLabel}`,
        action: c.action,
        playerId: c.playerId,
        start: Math.round(startSec * 100) / 100,
        end: Math.round(endSec * 100) / 100,
        frameStart: c.frameStart,
        frameEnd: c.frameEnd,
        peakConf: Math.round(c.peakConf * 100) / 100,
      };
    });
 
  return clips;
}
 
/**
 * Convenience: load a JSON file of actions and return clips.
 *
 * Usage:
 *   const clips = await loadActionClips("/actions.json", { fps: 30 });
 */
export async function loadActionClips(url, opts = {}) {
  const res = await fetch(url);
  const raw = await res.json();
  return clusterActions(raw, opts);
}