import { useState, useMemo } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";

/**
 * BallSpeedChart
 *
 * Computes ball speed from per-frame position data and renders
 * a time-synced area chart. Filters low-confidence detections
 * to avoid spurious jumps, applies moving-average smoothing.
 *
 * Props:
 *   annotations  — the parsed annotation array (from annotationsRef.current)
 *   currentTime  — current video playback time in seconds
 *   fps          — video frame rate (default 60)
 *   onSeek       — callback(timeInSeconds) when user clicks the chart
 */

function BallSpeedChart({
  annotations,
  currentTime = 0,
  fps = 60,
  onSeek,
}) {
  const [confThreshold, setConfThreshold] = useState(0.1);
  const [smoothWindow, setSmoothWindow] = useState(5);

  // ── Compute speed data from annotations ────────────────────
  const speedData = useMemo(() => {
    if (!annotations || annotations.length === 0) return [];

    // Extract ball positions, filtering by confidence
    const points = [];
    for (const f of annotations) {
      const ball = f.ball;
      if (!ball) {
        points.push({ frame: f.frame, x: null, y: null, conf: 0 });
        continue;
      }
      const conf = ball.conf || 0;
      let x, y;
      if (ball.center) {
        [x, y] = ball.center;
      } else if (ball.x !== undefined) {
        x = ball.x;
        y = ball.y;
      } else {
        points.push({ frame: f.frame, x: null, y: null, conf: 0 });
        continue;
      }

      // Filter: low confidence or off-screen
      if (conf < confThreshold || x < 0 || y < 0) {
        points.push({ frame: f.frame, x: null, y: null, conf });
        continue;
      }
      points.push({ frame: f.frame, x, y, conf });
    }

    // Compute raw speed (Euclidean distance between consecutive valid points)
    const rawSpeed = new Array(points.length).fill(0);
    let prevValid = null;
    for (let i = 0; i < points.length; i++) {
      if (points[i].x === null) {
        rawSpeed[i] = 0;
        continue;
      }
      if (prevValid !== null) {
        const dt = points[i].frame - points[prevValid].frame;
        if (dt > 0 && dt <= 3) {
          const dx = points[i].x - points[prevValid].x;
          const dy = points[i].y - points[prevValid].y;
          rawSpeed[i] = Math.sqrt(dx * dx + dy * dy) / dt;
        }
      }
      prevValid = i;
    }

    // Moving average smoothing
    const smoothed = new Array(points.length).fill(0);
    const half = Math.floor(smoothWindow / 2);
    for (let i = 0; i < points.length; i++) {
      let sum = 0;
      let count = 0;
      for (let j = Math.max(0, i - half); j <= Math.min(points.length - 1, i + half); j++) {
        sum += rawSpeed[j];
        count++;
      }
      smoothed[i] = sum / count;
    }

    // Build chart data
    return points.map((p, i) => ({
      frame: p.frame,
      time: +(p.frame / fps).toFixed(3),
      speed: +smoothed[i].toFixed(1),
      rawSpeed: +rawSpeed[i].toFixed(1),
      conf: p.conf,
    }));
  }, [annotations, confThreshold, smoothWindow, fps]);

  const maxSpeed = useMemo(
    () => Math.max(40, ...speedData.map((d) => d.speed)),
    [speedData]
  );

  if (speedData.length === 0) {
    return (
      <div className="bg-neutral-900 rounded-lg border border-neutral-800 p-6 flex justify-center">
        <span className="text-xs text-neutral-500">No ball data available</span>
      </div>
    );
  }

  return (
    <div className="bg-neutral-900 rounded-lg border border-neutral-800 pt-3 px-3.5 pb-2">
      {/* Header */}
      <div className="flex items-baseline gap-2 mb-1.5">
        <span className="text-sm font-semibold text-neutral-300 tracking-wide">
          Ball Speed
        </span>
        <span className="text-xs text-neutral-500 font-mono">px/frame</span>
      </div>

      {/* Chart */}
      <div className="-mx-1.5">
        <ResponsiveContainer width="100%" height={140}>
          <AreaChart
            data={speedData}
            margin={{ top: 4, right: 8, bottom: 0, left: -20 }}
            onClick={(e) => {
              if (e?.activePayload?.[0]?.payload?.time !== undefined && onSeek) {
                onSeek(e.activePayload[0].payload.time);
              }
            }}
            style={{ cursor: onSeek ? "crosshair" : "default" }}
          >
            <defs>
              <linearGradient id="speedGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#FFDC28" stopOpacity={0.5} />
                <stop offset="100%" stopColor="#FFDC28" stopOpacity={0.03} />
              </linearGradient>
            </defs>

            <XAxis
              dataKey="time"
              tick={{ fontSize: 10, fill: "#666" }}
              tickFormatter={(v) => `${v.toFixed(1)}s`}
              axisLine={{ stroke: "#333" }}
              tickLine={false}
              interval="preserveStartEnd"
              minTickGap={40}
            />
            <YAxis
              domain={[0, maxSpeed]}
              tick={{ fontSize: 10, fill: "#666" }}
              axisLine={false}
              tickLine={false}
              tickCount={4}
            />

            <Tooltip
              contentStyle={{
                backgroundColor: "#1a1a1a",
                border: "1px solid #333",
                borderRadius: 6,
                fontSize: 11,
                color: "#ccc",
              }}
              labelFormatter={(v) => `${v}s`}
              formatter={(value, name) => [
                `${value} px/f`,
                name === "speed" ? "Smoothed" : "Raw",
              ]}
            />

            <Area
              type="monotone"
              dataKey="speed"
              stroke="#FFDC28"
              strokeWidth={1.5}
              fill="url(#speedGrad)"
              dot={false}
              animationDuration={0}
            />

            {/* Playhead */}
            <ReferenceLine
              x={+currentTime.toFixed(3)}
              stroke="#fff"
              strokeWidth={1.5}
              strokeDasharray="3 2"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Controls */}
      <div className="flex gap-4 mt-1.5 pt-1.5 border-t border-neutral-800">
        <div className="flex items-center gap-1.5">
          <label className="text-xs text-neutral-500 font-mono whitespace-nowrap min-w-[90px]">
            Min conf: {confThreshold.toFixed(2)}
          </label>
          <input
            type="range"
            min={0}
            max={0.5}
            step={0.01}
            value={confThreshold}
            onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
            className="w-[70px] accent-yellow-400"
          />
        </div>
        <div className="flex items-center gap-1.5">
          <label className="text-xs text-neutral-500 font-mono whitespace-nowrap">
            Smooth: {smoothWindow}
          </label>
          <input
            type="range"
            min={1}
            max={15}
            step={2}
            value={smoothWindow}
            onChange={(e) => setSmoothWindow(parseInt(e.target.value))}
            className="w-[70px] accent-yellow-400"
          />
        </div>
      </div>
    </div>
  );
}

export default BallSpeedChart;