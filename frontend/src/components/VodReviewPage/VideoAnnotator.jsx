import { useState, useRef, useEffect, useCallback } from "react";
import Toggle from "./VideoPlayer/Toggle";

// ── Color palette for tracker IDs ──────────────────────────
const PLAYER_COLORS = [
  "#FF6B6B","#4ECDC4","#45B7D1","#96CEB4","#FFEAA7",
  "#DDA0DD","#98D8C8","#F7DC6F","#BB8FCE","#85C1E9",
  "#F0B27A","#82E0AA","#F1948A","#AED6F1","#D7BDE2",
  "#A3E4D7","#FAD7A0","#A9CCE3","#D5F5E3","#FADBD8",
];

function getPlayerColor(tid) {
  return PLAYER_COLORS[tid % PLAYER_COLORS.length];
}

// ── Main component ─────────────────────────────────────────

function VideoAnnotator({ url, annotationUrl, activeClip }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const animFrameRef = useRef(null);
  const annotationsRef = useRef([]);

  const [loaded, setLoaded] = useState(false);
  const [showPlayers, setShowPlayers] = useState(true);
  const [showBall, setShowBall] = useState(true);
  const [showTrail, setShowTrail] = useState(false);
  const [showLabels, setShowLabels] = useState(true);
  const [minConf, setMinConf] = useState(0.5);
  const [videoSize, setVideoSize] = useState({ w: 1920, h: 1080 });

  useEffect(() => {
    if (activeClip && videoRef.current) {
      videoRef.current.currentTime = activeClip.start;
      videoRef.current.play();
    }
  }, [activeClip]);

  // ── Load JSONL annotations ────────────────────────────────
  useEffect(() => {
    if (!annotationUrl) return;
    fetch(annotationUrl)
      .then((r) => r.text())
      .then((text) => {
        const frames = text
          .trim()
          .split("\n")
          .map((line) => JSON.parse(line));
        annotationsRef.current = frames;
        setLoaded(true);
      })
      .catch((err) => console.error("Failed to load annotations:", err));
  }, [annotationUrl]);

  // ── Sync canvas size to video's displayed size ────────────
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const syncSize = () => {
      const rect = video.getBoundingClientRect();
      const canvas = canvasRef.current;
      if (canvas && rect.width > 0) {
        canvas.width = rect.width;
        canvas.height = rect.height;
        canvas.style.width = rect.width + "px";
        canvas.style.height = rect.height + "px";
      }
    };

    const onMeta = () => {
      setVideoSize({ w: video.videoWidth, h: video.videoHeight });
      syncSize();
    };

    video.addEventListener("loadedmetadata", onMeta);
    window.addEventListener("resize", syncSize);
    const resizeObs = new ResizeObserver(syncSize);
    resizeObs.observe(video);

    return () => {
      video.removeEventListener("loadedmetadata", onMeta);
      window.removeEventListener("resize", syncSize);
      resizeObs.disconnect();
    };
  }, []);

  // ── Render loop: draw annotations on canvas each frame ────
  const draw = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !loaded) {
      animFrameRef.current = requestAnimationFrame(draw);
      return;
    }

    const ctx = canvas.getContext("2d");
    const cw = canvas.width;
    const ch = canvas.height;
    ctx.clearRect(0, 0, cw, ch);

    // Map current video time → frame index
    // Bail if video duration isn't ready yet
    const duration = video.duration;
    if (!duration || !isFinite(duration)) {
      animFrameRef.current = requestAnimationFrame(draw);
      return;
    }

    const totalFrames = annotationsRef.current.length;
    const fps = totalFrames / duration;
    const frameIdx = Math.min(
      Math.max(0, Math.round(video.currentTime * fps)),
      totalFrames - 1
    );
    const data = annotationsRef.current[frameIdx];
    if (!data) {
      animFrameRef.current = requestAnimationFrame(draw);
      return;
    }

    // Scale factors: annotation pixel coords → canvas coords
    // The <video> element includes the controls bar, so the actual
    // picture area is smaller than the element. Compute the picture
    // rect using the intrinsic aspect ratio.
    const videoAspect = videoSize.w / videoSize.h;
    const elemAspect = cw / ch;

    let picW, picH, offsetX, offsetY;
    if (elemAspect > videoAspect) {
      // element is wider than video — pillarboxed (black bars on sides)
      picH = ch;
      picW = ch * videoAspect;
      offsetX = (cw - picW) / 2;
      offsetY = 0;
    } else {
      // element is taller than video — letterboxed (black bar at bottom, i.e. controls)
      picW = cw;
      picH = cw / videoAspect;
      offsetX = 0;
      offsetY = 0; // video picture sits at top, controls fill the gap below
    }

    const sx = picW / videoSize.w;
    const sy = picH / videoSize.h;

    // ── Ball trail (last N positions) ────────────────────────
    if (showTrail) {
      const trailLen = 30;
      const startIdx = Math.max(0, frameIdx - trailLen);
      const trailPts = [];
      for (let i = startIdx; i <= frameIdx; i++) {
        const f = annotationsRef.current[i];
        if (f?.ball && f.ball.conf > 0 && f.ball.x >= 0 && f.ball.y >= 0) {
          trailPts.push({ x: f.ball.x * sx + offsetX, y: f.ball.y * sy + offsetY, age: frameIdx - i });
        }
      }
      if (trailPts.length > 1) {
        for (let i = 1; i < trailPts.length; i++) {
          const alpha = 1 - trailPts[i].age / trailLen;
          const width = 1 + 2 * alpha;
          ctx.beginPath();
          ctx.moveTo(trailPts[i - 1].x, trailPts[i - 1].y);
          ctx.lineTo(trailPts[i].x, trailPts[i].y);
          ctx.strokeStyle = `rgba(255, 220, 40, ${alpha * 0.9})`;
          ctx.lineWidth = width;
          ctx.stroke();
        }
      }
    }

    // ── Player bounding boxes ────────────────────────────────
    if (showPlayers && data.players) {
      for (const p of data.players) {
        if (p.conf < minConf) continue;
        const [x1, y1, x2, y2] = p.box;
        const bx = x1 * sx + offsetX;
        const by = y1 * sy + offsetY;
        const bw = (x2 - x1) * sx;
        const bh = (y2 - y1) * sy;
        const color = getPlayerColor(p.tid);

        // Box
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(bx, by, bw, bh);

        // Fill
        ctx.fillStyle = color + "18";
        ctx.fillRect(bx, by, bw, bh);

        // Label
        if (showLabels) {
          const label = `#${p.tid}`;
          const fontSize = Math.max(10, Math.min(14, bw * 0.35));
          ctx.font = `600 ${fontSize}px ui-monospace, monospace`;
          const metrics = ctx.measureText(label);
          const lw = metrics.width + 8;
          const lh = fontSize + 6;

          ctx.fillStyle = color;
          ctx.fillRect(bx, by - lh - 2, lw, lh);

          ctx.fillStyle = "#000";
          ctx.textBaseline = "top";
          ctx.fillText(label, bx + 4, by - lh + 1);
        }
      }
    }

    // ── Ball marker ──────────────────────────────────────────
    if (showBall && data.ball && data.ball.conf > 0 && data.ball.x >= 0 && data.ball.y >= 0) {
      const bx = data.ball.x * sx + offsetX;
      const by = data.ball.y * sy + offsetY;
      const radius = 8;

      // Outer glow
      const grad = ctx.createRadialGradient(bx, by, 0, bx, by, radius * 2.5);
      grad.addColorStop(0, "rgba(255, 220, 40, 0.6)");
      grad.addColorStop(1, "rgba(255, 220, 40, 0)");
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(bx, by, radius * 2.5, 0, Math.PI * 2);
      ctx.fill();

      // Inner circle
      ctx.fillStyle = "#FFDC28";
      ctx.strokeStyle = "#000";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(bx, by, radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }

    animFrameRef.current = requestAnimationFrame(draw);
  }, [loaded, showPlayers, showBall, showTrail, showLabels, minConf, videoSize]);

  useEffect(() => {
    animFrameRef.current = requestAnimationFrame(draw);
    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [draw]);

  return (
    <div className="flex flex-col gap-3 w-full max-w-6xl">
      {/* ── Video + Canvas overlay ── */}
      <div className="relative inline-block bg-black rounded-lg overflow-hidden shadow-xl">
        <video
          ref={videoRef}
          className="block w-full"
          controls
          playsInline
        >
          <source src={url} type="video/mp4" />
          Sorry, your browser doesn't support videos.
        </video>
        <canvas
          ref={canvasRef}
          className="absolute top-0 left-0 pointer-events-none"
          style={{ width: "100%", height: "100%" }}
        />
      </div>

      {/* ── Controls ── */}
      <div className="flex flex-wrap items-center gap-2 px-1">
        <Toggle on={showPlayers} onToggle={() => setShowPlayers((v) => !v)} label="Players" color="#4ECDC4" />
        <Toggle on={showBall} onToggle={() => setShowBall((v) => !v)} label="Ball" color="#FFDC28" />
        <Toggle on={showTrail} onToggle={() => setShowTrail((v) => !v)} label="Trail" color="#F0B27A" />
        <Toggle on={showLabels} onToggle={() => setShowLabels((v) => !v)} label="Labels" color="#BB8FCE" />

        <div className="ml-auto flex items-center gap-2 text-sm text-gray-400">
          <span className="font-mono text-xs">conf ≥ {minConf.toFixed(2)}</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={minConf}
            onChange={(e) => setMinConf(parseFloat(e.target.value))}
            className="w-24 accent-teal-400"
          />
        </div>
      </div>

      {!loaded && annotationUrl && (
        <div className="text-sm text-gray-500 px-1">Loading annotations…</div>
      )}
    </div>
    
    
  );
  
}

export default VideoAnnotator;