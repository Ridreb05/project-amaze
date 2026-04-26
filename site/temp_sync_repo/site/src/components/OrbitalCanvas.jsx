import { useEffect, useRef } from 'react';

function mulberry32(seed) {
  return function() {
    let t = seed += 0x6D2B79F5;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

export function OrbitalCanvas({ data, episodeIdx, stepIdx }) {
  const canvasRef = useRef(null);

  const visualPosRef = useRef({ x: 0, y: 0 });
  const requestRef = useRef();

  const drawSpacecraft = (ctx, x, y, angle, color, label) => {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(angle);
    
    // Solar Panels
    ctx.fillStyle = '#1e293b';
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.fillRect(-15, -2, 30, 4); // Main panel spread
    ctx.strokeRect(-15, -2, 30, 4);
    
    // Panel Grid lines
    ctx.strokeStyle = 'rgba(255,255,255,0.1)';
    for(let i=-12; i<=12; i+=6) {
      ctx.beginPath(); ctx.moveTo(i, -2); ctx.lineTo(i, 2); ctx.stroke();
    }

    // Body
    ctx.fillStyle = '#f8fafc';
    ctx.beginPath();
    ctx.roundRect(-4, -6, 8, 12, 2);
    ctx.fill();
    ctx.strokeStyle = color;
    ctx.stroke();

    // Antenna/Detail
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(0, -3, 1.5, 0, Math.PI*2); ctx.fill();

    ctx.restore();

    // Label
    ctx.fillStyle = color;
    ctx.font = 'bold 10px var(--font)';
    ctx.textAlign = 'left';
    ctx.fillText(label, x + 18, y + 4);
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    // Fixed Sizing: Only run once or on Resize to prevent layout jitter
    const dpr = window.devicePixelRatio || 1;
    const displaySize = canvas.parentElement.offsetWidth || 400;
    
    canvas.width = displaySize * dpr;
    canvas.height = displaySize * dpr;
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    ctx.scale(dpr, dpr);

    const width = displaySize;
    const height = displaySize;

    const animate = () => {
      // Background
      ctx.fillStyle = 'rgba(10, 14, 26, 1)';
      ctx.fillRect(0, 0, width, height);
      
      const rng = mulberry32(12345);
      ctx.fillStyle = 'rgba(255,255,255,0.4)';
      for (let i = 0; i < 80; i++) {
          ctx.beginPath();
          ctx.arc(rng() * width, rng() * height, rng() * 1.2, 0, Math.PI * 2);
          ctx.fill();
      }

      if (!data || !data.episodes[episodeIdx]) {
        ctx.fillStyle = '#94a3b8'; ctx.font = '13px var(--font)'; ctx.textAlign = 'center';
        ctx.fillText('Load training_log.json to begin', width/2, height/2);
        return;
      }

      const ep = data.episodes[episodeIdx];
      const step = ep.trajectory[stepIdx] || ep.trajectory[ep.trajectory.length - 1];
      if (!step) return;

      // Bug Fix: Reset visual position on first step of a new episode to prevent 'sliding' from previous episode coords
      if (stepIdx === 0) {
        visualPosRef.current = { x: step.y_m, y: step.x_m };
      }

      // Lerp Visual Position for smoothness
      const lerpFactor = 0.15;
      visualPosRef.current.x += (step.y_m - visualPosRef.current.x) * lerpFactor;
      visualPosRef.current.y += (step.x_m - visualPosRef.current.y) * lerpFactor;

      const maxDist = Math.max(...ep.trajectory.map(s => Math.sqrt(s.x_m**2 + s.y_m**2)), ep.initial_distance_m, 10);
      const scale = (width / 2 * 0.85) / maxDist;
      const cx = width / 2;
      const cy = height / 2;

      // Grid
      const gridSpacing = [10, 25, 50, 100, 200].find(s => s * scale > 30) || 200;
      ctx.strokeStyle = 'rgba(42,58,85,0.4)'; ctx.lineWidth = 0.5;
      for (let d = -6; d <= 6; d++) {
        const x = cx + d * gridSpacing * scale;
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
        const y = cy + d * gridSpacing * scale;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
      }

      // Cone
      const coneRange = 50 * scale;
      const coneAngle = Math.PI / 4;
      ctx.fillStyle = 'rgba(16,185,129,0.06)';
      ctx.beginPath(); ctx.moveTo(cx, cy);
      ctx.arc(cx, cy, coneRange, -Math.PI/2 - coneAngle, -Math.PI/2 + coneAngle);
      ctx.fill();

      // Trail
      const trail = ep.trajectory.slice(0, stepIdx + 1);
      if (trail.length > 1) {
        ctx.beginPath(); ctx.strokeStyle = ep.docked ? 'rgba(16,185,129,0.3)' : 'rgba(239,68,68,0.2)';
        ctx.lineWidth = 1.5;
        trail.forEach((s, i) => {
          const px = cx + s.y_m * scale;
          const py = cy - s.x_m * scale;
          i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
        });
        ctx.stroke();
      }

      // Target
      drawSpacecraft(ctx, cx, cy, 0, '#f59e0b', 'TARGET');

      // Chaser
      const px = cx + visualPosRef.current.x * scale;
      const py = cy - visualPosRef.current.y * scale;
      const docked = step.distance_m < 1.1;
      const color = docked ? '#10b981' : (step.los_violation ? '#ef4444' : '#3b82f6');
      
      // Thrust
      const mag = Math.sqrt(step.fx**2 + step.fy**2);
      if (mag > 0.02) {
        const tx = step.fy * 20, ty = -step.fx * 20;
        ctx.strokeStyle = 'rgba(245,158,11,0.6)'; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(px - tx, py - ty); ctx.stroke();
      }

      drawSpacecraft(ctx, px, py, Math.atan2(step.y_m, step.x_m), color, 'CHASER');

      // Distance Line
      ctx.beginPath(); ctx.strokeStyle = 'rgba(148,163,184,0.1)'; ctx.setLineDash([4, 4]);
      ctx.moveTo(cx, cy); ctx.lineTo(px, py); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = '#94a3b8'; ctx.font = '9px var(--font)';
      ctx.fillText(`${step.distance_m.toFixed(1)}m`, (cx + px)/2 + 8, (cy + py)/2);

      // HUD
      ctx.fillStyle = 'rgba(148,163,184,0.8)'; ctx.textAlign = 'left';
      ctx.fillText(`EP ${ep.episode} | STEP ${stepIdx + 1}`, 10, 20);
      ctx.fillText(`DIST: ${step.distance_m.toFixed(2)}m`, 10, 34);

      // Mission End Overlay - Show whenever simulation reaches the end of the trajectory
      if (stepIdx >= ep.trajectory.length - 1) {
        // Semi-transparent overlay to make status clear
        ctx.fillStyle = ep.docked ? 'rgba(16,185,129,0.1)' : 'rgba(239,68,68,0.08)';
        ctx.fillRect(0, 0, width, height);
        
        ctx.fillStyle = ep.docked ? '#10b981' : '#ef4444';
        ctx.font = 'bold 20px var(--font)';
        ctx.textAlign = 'center';
        ctx.fillText(ep.docked ? 'DOCKING SUCCESSFUL' : 'MISSION FAILED', width/2, height/2 - 10);
        
        ctx.font = '12px var(--font)';
        ctx.fillStyle = 'rgba(255,255,255,0.7)';
        ctx.fillText(ep.docked ? 'SOFT CAPTURE CONFIRMED' : 'TERMINATED - DATA RECOVERY IDLE', width/2, height/2 + 15);
        ctx.textAlign = 'left'; // Reset
      }

      requestRef.current = requestAnimationFrame(animate);
    };

    requestRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(requestRef.current);
  }, [data, episodeIdx, stepIdx]);

  return (
    <div className="panel">
      <div className="panel-title">Orbital View — LVLH Frame</div>
      <div className="orbital-container">
        <canvas id="orbital-canvas" ref={canvasRef}></canvas>
      </div>
    </div>
  );
}
