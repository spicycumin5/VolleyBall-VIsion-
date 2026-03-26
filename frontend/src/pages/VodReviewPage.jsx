import { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import VideoPlayer from "../components/VodReviewPage/VideoPlayer";
import Playlist from "../components/VodReviewPage/Playlist";
import "./VodReviewPage.css";
 
/**
 * VodReviewPage.jsx
 *
 * Reads video info from React Router location.state (passed by SessionPicker).
 * Calls detectClips() to get AI-parsed clip timestamps, then renders
 * the video player and clip playlist side by side.
 *
 * Expected location.state shape:
 *   { videoSrc: string, videoName: string }
 */
 
export default function VodReviewPage() {
  const location = useLocation();
  const navigate = useNavigate();
 
  const { videoSrc, videoName } = location.state || {};
 
  const [clips, setClips] = useState([]);
  const [activeIndex, setActiveIndex] = useState(null);
  const [loading, setLoading] = useState(true);
 
  useEffect(() => {
    if (!videoSrc) return;
 
    setLoading(true);
    detectClips(videoSrc)
      .then((detected) => {
        setClips(detected);
        setActiveIndex(0);
      })
      .finally(() => setLoading(false));
  }, [videoSrc]);
 
  if (!videoSrc) {
    return (
      <div className="vod-error">
        <p>No session selected.</p>
        <button onClick={() => navigate("/")}>Back to sessions</button>
      </div>
    );
  }
 
  const activeClip = activeIndex !== null ? clips[activeIndex] : null;
 
  return (
    <div className="vod">
 
      <div className="vod__topbar">
        <button className="vod__back" onClick={() => navigate("/")}>← Sessions</button>
        <h1 className="vod__title">{videoName}</h1>
      </div>
 
      <div className="vod__body">
 
        <div className="vod__player">
          <VideoPlayer src={videoSrc} activeClip={activeClip} />
        </div>
 
        <div className="vod__sidebar">
          {loading ? (
            <div className="vod__loading">
              <p>Detecting clips…</p>
            </div>
          ) : (
            <Playlist
              clips={clips}
              activeIndex={activeIndex}
              onSelect={setActiveIndex}
            />
          )}
        </div>
 
      </div>
    </div>
  );
}