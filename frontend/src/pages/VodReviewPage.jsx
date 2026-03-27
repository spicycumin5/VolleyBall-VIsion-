import { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { loadActionClips } from "../components/VodReviewPage/clusterActions";

import VideoAnnotator from "../components/VodReviewPage/VideoAnnotator";
import Playlist from "../components/VodReviewPage/Playlist";
import SESSIONS from "../assets/videos/sessions";
import "./VodReviewPage.css";

export default function VodReviewPage() {
  const location = useLocation();
  const navigate = useNavigate();
 
  const { sessionKey } = location.state || {};
  const session = SESSIONS.find((s) => s.key === sessionKey);
 
  const [clips, setClips] = useState([]);
  const [activeIndex, setActiveIndex] = useState(0);
 
  // Load and cluster action detections into playlist clips
  useEffect(() => {
    if (!session?.actionURL) return;
    loadActionClips(session.actionURL, {
      fps: 60,
      minConf: 0.01,
      gapFrames: 10,
      padSeconds: 1,
      minClipFrames: 1,
    }).then(setClips);
  }, [session?.actionURL]);
 
  if (!session) {
    return (
      <div className="vod-error">
        <p>No session selected.</p>
        <button onClick={() => navigate("/home")}>Back to sessions</button>
      </div>
    );
  }
 
  const activeClip = clips[activeIndex] ?? null;
 
  return (
    <div className="vod">
      <div className="vod__topbar">
        <button className="vod__back" onClick={() => navigate("/home")}>← Sessions</button>
        <h1 className="vod__title">{session.title}</h1>
      </div>
 
      <div className="vod__body">
        <div className="vod__player">
          <VideoAnnotator
            url={session.videoSrc}
            annotationUrl={session.annotationUrl}
            activeClip={activeClip}
          />
        </div>
 
        <div className="vod__sidebar">
          <Playlist
            clips={clips}
            activeIndex={activeIndex}
            onSelect={setActiveIndex}
          />
        </div>
      </div>
    </div>
  );
}