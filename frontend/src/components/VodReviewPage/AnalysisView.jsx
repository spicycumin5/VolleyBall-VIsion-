import { useState, useEffect } from "react";
import VideoAnnotator from "./VideoAnnotator";
import Playlist from "./Playlist";
import { loadActionClips } from "./clusterActions";

export default function AnalysisView({ videoUrl, annotationUrl, actionsUrl }) {
  const [clips, setClips] = useState([]);
  const [activeIndex, setActiveIndex] = useState(-1);

  // Load and cluster action detections into playlist clips
  useEffect(() => {
    if (!actionsUrl) return;
    loadActionClips(actionsUrl, {
      fps: 60,        // match your video's fps
      minConf: 0.25,   // filter out noise below 30% confidence
      gapFrames: 10,  // merge detections within 10 frames of each other
      padSeconds: 1,  // add 1s of context before/after each clip
    }).then(setClips);
  }, [actionsUrl]);

  const activeClip = activeIndex >= 0 ? clips[activeIndex] : null;

  return (
    <div className="flex gap-4">
      <VideoAnnotator
        url={videoUrl}
        annotationUrl={annotationUrl}
        activeClip={activeClip}
      />
      <Playlist
        clips={clips}
        activeIndex={activeIndex}
        onSelect={setActiveIndex}
      />
    </div>
  );
}

/**
 * Usage:
 *
 *   <AnalysisView
 *     videoUrl="/practice.mp4"
 *     annotationUrl="/one_play_superresolution.json"
 *     actionsUrl="/actions.json"          ← your raw action detections
 *   />
 *
 * Put actions.json in public/ just like the other annotation files.
 *
 * The Playlist component doesn't need any changes — it already
 * expects { id, title, start, end } which is exactly what
 * clusterActions produces.
 */
