import { useEffect, useRef } from "react";
import "./VideoPlayer.css";
 
/**
 * VideoPlayer.jsx
 *
 * Plays a video and seeks to a clip's start time when `activeClip` changes.
 * Pauses automatically when the clip's end time is reached.
 */
 
const formatTime = (s) => {
  if (!s || isNaN(s)) return "0:00";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
};
 
export default function VideoPlayer({ src, activeClip }) {
  const videoRef = useRef(null);
 
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !activeClip) return;
 
    video.currentTime = activeClip.start;
    video.play().catch(() => {});
 
    const handleTimeUpdate = () => {
      if (video.currentTime >= activeClip.end) {
        video.pause();
        video.removeEventListener("timeupdate", handleTimeUpdate);
      }
    };
 
    video.addEventListener("timeupdate", handleTimeUpdate);
    return () => video.removeEventListener("timeupdate", handleTimeUpdate);
  }, [activeClip]);
 
  return (
    <div className="vplayer">
      <div className="vplayer__screen">
        {src ? (
          <video
            ref={videoRef}
            src={src}
            className="vplayer__video"
            controls
          />
        ) : (
          <div className="vplayer__empty">No video loaded</div>
        )}
      </div>
 
      {activeClip && (
        <div className="vplayer__clip-bar">
          <span className="vplayer__clip-label">{activeClip.label}</span>
          <span className="vplayer__clip-time">
            {formatTime(activeClip.start)} – {formatTime(activeClip.end)}
          </span>
        </div>
      )}
    </div>
  );
}