  import { useState, useEffect } from "react";
  import { useParams, useLocation, useNavigate } from "react-router-dom";
  import VideoAnnotator from "../components/VodReviewPage/VideoAnnotator";
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

  //mock ai clip detection function
  async function detectClips(videoSrc) {
    // Simulate network delay
    await new Promise((resolve) => setTimeout(resolve, 1500));

    return [
      { id: 1, title: "Serve", start: 10, end: 45 },
      { id: 2, title: "Spike", start: 120, end: 180 },
      { id: 3, title: "Block", start: 300, end: 340 },
    ];
  }

  const placeholderVideo = "/videos/one_play.mp4";
  const placeholderJSON = "/annotations/one_play.json";

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
          <button className="vod__back" onClick={() => navigate("/home")}>← Sessions</button>
          <h1 className="vod__title">{videoName}</h1>
        </div>
  
        <div className="vod__body">
  
          <div className="vod__player">
            <VideoAnnotator url={placeholderVideo} 
            annotationUrl={placeholderJSON} 
            activeClip={activeClip}/>
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