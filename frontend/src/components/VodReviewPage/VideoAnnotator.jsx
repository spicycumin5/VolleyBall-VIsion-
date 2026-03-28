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

// ── Extract ball position from annotation ──────────────────
// Format: { tid, box: [x1,y1,x2,y2], center: [x,y], conf, predicted }
// ball may be null — returns null in that case.
function getBallPos(ball) {
  if (!ball || ball.conf <= 0) return null;
  if (!ball.center) return null;
  const [x, y] = ball.center;
  if (x < 0 || y < 0) return null;
  return {
    x,
    y,
    box: ball.box ?? null,
    conf: ball.conf,
    predicted: ball.predicted ?? false,
  };
}

// ── Main component ─────────────────────────────────────────

function VideoAnnotator({ url, annotationUrl, activeClip, onTimeUpdate, onAnnotationsLoaded }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const animFrameRef = useRef(null);
  const annotationsRef = useRef([]);

  const [loaded, setLoaded] = useState(false);
  const [showPlayers, setShowPlayers] = useState(true);
  const [showBall, setShowBall] = useState(true);
  const [showTrail, setShowTrail] = useState(true);
  const [showLabels, setShowLabels] = useState(true);
  const [minConf, setMinConf] = useState(0.5);
  const [videoSize, setVideoSize] = useState({ w: 1920, h: 1080 });

  useEffect(() => {
    if (activeClip && videoRef.current) {
      videoRef.current.currentTime = activeClip.start;
      videoRef.current.play();
    }
  }, [activeClip]);

  // ── Load JSON annotations ─────────────────────────────────
  // Format: a JSON array of frame objects, each with { frame, ball, players }
  useEffect(() => {
    if (!annotationUrl) return;
    fetch(annotationUrl)
      .then((r) => r.json())
      .then((frames) => {
        // Sort by frame number to guarantee correct temporal order
        const sorted = [...frames].sort((a, b) => a.frame - b.frame);
        annotationsRef.current = sorted;
        setLoaded(true);
        if (onAnnotationsLoaded) onAnnotationsLoaded(sorted);
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

    const duration = video.duration;
    if (!duration || !isFinite(duration)) {
      animFrameRef.current = requestAnimationFrame(draw);
      return;
    }

    if (onTimeUpdate) onTimeUpdate(video.currentTime);

    const frames = annotationsRef.current;
    const totalFrames = frames.length;
    const fps = totalFrames / duration;

    // Map current time → nearest frame index in the sorted array
    const targetFrame = Math.round(video.currentTime * fps);
    const frameIdx = Math.min(Math.max(0, targetFrame), totalFrames - 1);
    const data = frames[frameIdx];
    if (!data) {
      animFrameRef.current = requestAnimationFrame(draw);
      return;
    }

    // ── Compute picture rect ──────────────────────────────────
    // The <video> element may include a controls bar below the picture.
    // Derive the actual picture area from the intrinsic aspect ratio so
    // that all annotation coordinates map to the correct screen position.
    const videoAspect = videoSize.w / videoSize.h;
    const elemAspect = cw / ch;

    let picW, picH, offsetX, offsetY;
    if (elemAspect > videoAspect) {
      // Element wider than video → pillarboxed
      picH = ch;
      picW = ch * videoAspect;
      offsetX = (cw - picW) / 2;
      offsetY = 0;
    } else {
      // Element taller than video → letterboxed (controls at bottom)
      picW = cw;
      picH = cw / videoAspect;
      offsetX = 0;
      offsetY = 0;
    }

    const sx = picW / videoSize.w;
    const sy = picH / videoSize.h;

    // ── Ball trail ────────────────────────────────────────────
    if (showTrail) {
      const trailLen = 30;
      const startIdx = Math.max(0, frameIdx - trailLen);
      const trailPts = [];
      for (let i = startIdx; i <= frameIdx; i++) {
        const f = frames[i];
        const bp = getBallPos(f?.ball);
        if (bp) {
          trailPts.push({ x: bp.x * sx + offsetX, y: bp.y * sy + offsetY, age: frameIdx - i });
        }
      }
      if (trailPts.length > 1) {
        for (let i = 1; i < trailPts.length; i++) {
          const alpha = 1 - trailPts[i].age / trailLen;
          ctx.beginPath();
          ctx.moveTo(trailPts[i - 1].x, trailPts[i - 1].y);
          ctx.lineTo(trailPts[i].x, trailPts[i].y);
          ctx.strokeStyle = `rgba(0, 33, 255, ${alpha * 0.9})`;
          ctx.lineWidth = 1 + 2 * alpha;
          ctx.stroke();
        }
      }
    }

    // ── Player bounding boxes ─────────────────────────────────
    if (showPlayers && data.players) {
      for (const p of data.players) {
        if (p.conf < minConf) continue;
        const [x1, y1, x2, y2] = p.box;
        const bx = x1 * sx + offsetX;
        const by = y1 * sy + offsetY;
        const bw = (x2 - x1) * sx;
        const bh = (y2 - y1) * sy;
        const color = getPlayerColor(p.tid);

        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(bx, by, bw, bh);

        ctx.fillStyle = color + "18";
        ctx.fillRect(bx, by, bw, bh);

        if (showLabels) {
          const stateTag = p.state && p.state !== "player" ? ` ${p.state}` : "";
          const label = `#${p.tid}${stateTag}`;
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

    // ── Ball marker ───────────────────────────────────────────
    const ballPos = getBallPos(data.ball);
    if (showBall && ballPos) {
      if (ballPos.box) {
        const [x1, y1, x2, y2] = ballPos.box;
        const bx = x1 * sx + offsetX;
        const by = y1 * sy + offsetY;
        const bw = (x2 - x1) * sx;
        const bh = (y2 - y1) * sy;

        ctx.strokeStyle = ballPos.predicted ? "rgba(0, 33, 255, 0.5)" : "#0021ff";
        ctx.lineWidth = 2;
        if (ballPos.predicted) ctx.setLineDash([4, 3]);
        ctx.strokeRect(bx, by, bw, bh);
        ctx.setLineDash([]);

        ctx.fillStyle = ballPos.predicted
          ? "rgba(0, 33, 255, 0.06)"
          : "rgba(0, 33, 255, 0.12)";
        ctx.fillRect(bx, by, bw, bh);
      } else {
        // Fallback: crosshair at center point
        const cx = ballPos.x * sx + offsetX;
        const cy = ballPos.y * sy + offsetY;
        const arm = 6;
        ctx.strokeStyle = "#0021ff";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(cx - arm, cy);
        ctx.lineTo(cx + arm, cy);
        ctx.moveTo(cx, cy - arm);
        ctx.lineTo(cx, cy + arm);
        ctx.stroke();
      }
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