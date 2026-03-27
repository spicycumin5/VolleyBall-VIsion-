import { useState, useEffect } from "react";
  import { useParams, useLocation, useNavigate } from "react-router-dom";
  import VideoAnnotator from "../components/VodReviewPage/VideoAnnotator";
  import Playlist from "../components/VodReviewPage/Playlist";
  import "./VodReviewPage.css";
  import SESSIONS from "../assets/videos/sessions.js";

export default function VodReviewPage() {
  const location = useLocation();
  const navigate = useNavigate();

  
  const { sessionKey } = location.state || {};

  const session = SESSIONS.find((s) => s.key === sessionKey);
  
  const [activeIndex, setActiveIndex] = useState(0);

  if (!session) {
    return (
      <div className="vod-error">
        <p>No session selected.</p>
        <button onClick={() => navigate("/home")}>Back to sessions</button>
      </div>
    );
  }

  const activeClip = session.clips[activeIndex];

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
            clips={session.clips}
            activeIndex={activeIndex}
            onSelect={setActiveIndex}
          />
        </div>
      </div>
    </div>
  );
}