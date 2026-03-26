import "./Playlist.css";
 
/**
 * Playlist.jsx
 *
 * Displays a scrollable list of AI-detected clips.
 * Each row shows the clip label and its timestamp range.
 *
 * Props:
 *   clips       — array of { id, label, start, end } from clipDetection.js
 *   activeIndex — index of the currently selected clip
 *   onSelect(i) — called when a row is clicked
 */
 
const formatTime = (s) => {
  if (!s || isNaN(s)) return "0:00";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
};
 
export default function Playlist({ clips, activeIndex, onSelect }) {
  return (
    <div className="playlist">
 
      <div className="playlist__header">
        <p className="playlist__title">
          Clips · {clips.length} detected
        </p>
      </div>
 
      <div className="playlist__list">
        {clips.length === 0 ? (
          <p className="playlist__empty">No clips detected</p>
        ) : (
          clips.map((clip, i) => (
            <div
              key={clip.id}
              onClick={() => onSelect(i)}
              className={`playlist__row ${i === activeIndex ? "playlist__row--active" : ""}`}
            >
              <span className="playlist__index">{i + 1}</span>
              <div className="playlist__meta">
                <span className="playlist__name">{clip.label}</span>
                <span className="playlist__time">
                  {formatTime(clip.start)} – {formatTime(clip.end)}
                </span>
              </div>
            </div>
          ))
        )}
      </div>
 
    </div>
  );
}