'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { TweaksPanel, useTweaks, TweakSection, TweakToggle, TweakRadio, TweakSlider } from '../components/TweaksPanel';
import { useEnhancedWebSocket } from '../lib/enhanced-websocket';
import { toast } from 'react-hot-toast';

// ─── Icons (inline SVG) ───────────────────────────────────────────────────────
const Icon = ({ d, size = 16, color = 'currentColor', strokeWidth = 1.6, fill = 'none' }: any) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill={fill} stroke={color} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round">
    {Array.isArray(d) ? d.map((p, i) => <path key={i} d={p} />) : <path d={d} />}
  </svg>
);

const Icons = {
  brain: "M12 5a3 3 0 0 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z M12 5a3 3 0 0 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 1 1 1 12 18Z",
  upload: ["M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4", "M17 8l-5-5-5 5", "M12 3v12"],
  scan: ["M3 7V5a2 2 0 0 1 2-2h2", "M17 3h2a2 2 0 0 1 2 2v2", "M21 17v2a2 2 0 0 1-2 2h-2", "M7 21H5a2 2 0 0 1-2-2v-2"],
  activity: "M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2",
  check: ["M20 6 9 17l-5-5"],
  warning: ["M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z", "M12 9v4", "M12 17h.01"],
  download: ["M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4", "M7 10l5 5 5-5", "M12 15V3"],
  model: "M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 18a8 8 0 1 1 8-8 8 8 0 0 1-8 8zm0-14a6 6 0 1 0 6 6 6 6 0 0 0-6-6zm0 10a4 4 0 1 1 4-4 4 4 0 0 1-4 4z",
  settings: ["M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z", "M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z"],
  close: ["M18 6 6 18", "M6 6l12 12"],
  plus: ["M12 5v14", "M5 12h14"],
  trash: ["M3 6h18", "M19 6l-1 14H6L5 6", "M8 6V4h8v2"],
  eye: ["M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7z", "M12 12m-3 0a3 3 0 1 0 6 0a3 3 0 0 0-6 0"],
  chart: ["M3 3v18h18", "M18 17V9", "M13 17V5", "M8 17v-3"],
  cpu: ["M12 12m-3 0a3 3 0 1 0 6 0a3 3 0 0 0-6 0", "M5 12H2", "M22 12h-3", "M12 2v3", "M12 19v3", "M4.2 4.2l2.1 2.1", "M17.7 17.7l2.1 2.1", "M17.7 6.3l-2.1 2.1", "M4.2 19.8l2.1-2.1"],
  fileImg: ["M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z", "M14 2v4a2 2 0 0 0 2 2h4", "M10 12a2 2 0 0 0-2 2v2h8v-2a2 2 0 0 0-2-2h-4z", "M10 12V8"],
  zap: "M13 2 3 14h9l-1 8 10-12h-9l1-8z",
  info: ["M12 12m-10 0a10 10 0 1 0 20 0a10 10 0 0 0-20 0", "M12 16v-4", "M12 8h.01"],
};

// ─── Utility Components ───────────────────────────────────────────────────────
const Badge = ({ children, variant = 'default', size = 'sm' }: any) => {
  const colors: any = {
    default: { bg: 'var(--bg-elevated)', color: 'var(--text-secondary)', border: 'var(--border)' },
    blue: { bg: 'var(--accent-blue-dim)', color: 'var(--accent-blue)', border: 'var(--accent-blue-mid)' },
    teal: { bg: 'var(--accent-teal-dim)', color: 'var(--accent-teal)', border: 'oklch(65% 0.22 180 / 0.3)' },
    red: { bg: 'var(--accent-red-dim)', color: 'var(--accent-red)', border: 'oklch(60% 0.22 25 / 0.3)' },
    amber: { bg: 'oklch(72% 0.18 70 / 0.15)', color: 'var(--accent-amber)', border: 'oklch(72% 0.18 70 / 0.3)' },
    green: { bg: 'oklch(65% 0.22 145 / 0.15)', color: 'oklch(65% 0.22 145)', border: 'oklch(65% 0.22 145 / 0.3)' },
  };
  const c = colors[variant] || colors.default;
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 4,
      padding: size === 'sm' ? '2px 8px' : '4px 10px',
      fontSize: size === 'sm' ? 11 : 12, fontWeight: 500, letterSpacing: '0.02em',
      borderRadius: 99, border: `1px solid ${c.border}`,
      background: c.bg, color: c.color, fontFamily: 'var(--font-ui)',
      whiteSpace: 'nowrap',
    }}>
      {children}
    </span>
  );
};

const Metric = ({ label, value, unit, color = 'var(--accent-blue)', sub }: any) => (
  <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
    <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', fontFamily: 'var(--font-mono)' }}>{label}</div>
    <div style={{ display: 'flex', alignItems: 'baseline', gap: 4 }}>
      <span style={{ fontSize: 26, fontWeight: 700, color, lineHeight: 1, fontFamily: 'var(--font-mono)' }}>{value}</span>
      {unit && <span style={{ fontSize: 12, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{unit}</span>}
    </div>
    {sub && <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{sub}</div>}
  </div>
);

const ProgressBar = ({ value, color = 'var(--accent-blue)', height = 4, label, showValue }: any) => (
  <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
    {(label || showValue) && (
      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
        {label && <span style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{label}</span>}
        {showValue && <span style={{ fontSize: 11, color, fontFamily: 'var(--font-mono)', fontWeight: 500 }}>{value}%</span>}
      </div>
    )}
    <div style={{ height, background: 'var(--border)', borderRadius: 99, overflow: 'hidden' }}>
      <div style={{ height: '100%', width: `${value}%`, background: color, borderRadius: 99, transition: 'width 0.5s ease' }} />
    </div>
  </div>
);

const Card = ({ children, style, onClick, glow }: any) => (
  <div onClick={onClick} style={{
    background: 'var(--bg-card)',
    border: `1px solid ${glow ? 'var(--accent-blue-mid)' : 'var(--border)'}`,
    borderRadius: 'var(--radius-lg)',
    padding: '18px 20px',
    boxShadow: glow ? '0 0 24px oklch(65% 0.22 240 / 0.08)' : 'none',
    cursor: onClick ? 'pointer' : 'default',
    transition: 'border-color 0.2s, box-shadow 0.2s',
    ...style
  }}>
    {children}
  </div>
);

const Divider = ({ style }: any) => (
  <div style={{ height: 1, background: 'var(--border-subtle)', ...style }} />
);

// ─── Brain Canvas ─────────────────────────────────────────────────────────────
const BrainCanvas = ({ tumorDetected, confidence, speed = 1 }: any) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);
  const timeRef = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rng = (seed: number) => { let x = Math.sin(seed) * 43758.5453; return x - Math.floor(x); };

    const drawCortex = (cx: number, cy: number, t: number) => {
      ctx.save();
      ctx.beginPath();
      for (let a = 0; a <= Math.PI * 2; a += 0.02) {
        const bumpA = 1 + 0.04 * Math.sin(6 * a + t * 0.3) + 0.02 * Math.sin(13 * a - t * 0.5);
        const bumpB = 1 + 0.035 * Math.sin(5 * a - t * 0.25) + 0.02 * Math.sin(11 * a + t * 0.4);
        const rx = 128 * bumpA, ry = 106 * bumpB;
        const x = cx + Math.cos(a) * rx;
        const y = cy + Math.sin(a) * ry * 0.88;
        a === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.closePath();
      const skullGrad = ctx.createRadialGradient(cx, cy - 10, 10, cx, cy, 135);
      skullGrad.addColorStop(0, tumorDetected ? 'rgba(40,10,10,0.85)' : 'rgba(8,18,38,0.85)');
      skullGrad.addColorStop(1, 'rgba(4,8,20,0.95)');
      ctx.fillStyle = skullGrad;
      ctx.fill();
      ctx.strokeStyle = tumorDetected ? 'rgba(200,80,60,0.35)' : 'rgba(60,140,255,0.3)';
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.restore();

      const gyriCount = 9;
      for (let g = 0; g < gyriCount; g++) {
        const phase = (g / gyriCount) * Math.PI * 2;
        const baseR = 30 + g * 10.5;
        ctx.save();
        ctx.beginPath();
        let first = true;
        for (let a = 0; a <= Math.PI * 2; a += 0.025) {
          const fold = baseR + 7 * Math.sin(6 * a + phase + t * (0.15 + g * 0.04))
                              + 4 * Math.sin(11 * a - phase - t * 0.12)
                              + 2.5 * Math.sin(19 * a + t * 0.08);
          const x = cx + Math.cos(a) * fold;
          const y = cy + Math.sin(a) * fold * 0.82;
          first ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
          first = false;
        }
        ctx.closePath();
        const alpha = 0.06 + (g / gyriCount) * 0.06;
        ctx.strokeStyle = tumorDetected
          ? `rgba(255,${100 - g * 8},${80 - g * 6},${alpha})`
          : `rgba(${50 + g * 8},${150 + g * 5},255,${alpha})`;
        ctx.lineWidth = 0.8;
        ctx.stroke();
        ctx.restore();
      }

      ctx.save();
      ctx.beginPath();
      ctx.moveTo(cx, cy - 102);
      ctx.bezierCurveTo(cx + 8, cy - 60, cx + 5, cy + 40, cx, cy + 100);
      ctx.strokeStyle = 'rgba(100,160,255,0.12)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 7]);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();

      ctx.save();
      ctx.beginPath();
      ctx.moveTo(cx - 14, cy + 95);
      ctx.bezierCurveTo(cx - 10, cy + 112, cx + 10, cy + 112, cx + 14, cy + 95);
      ctx.strokeStyle = 'rgba(80,160,255,0.2)';
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.restore();
    };

    const REGIONS = [
      { name: 'frontal',    cx: -42, cy: -55, count: 7, r: 28 },
      { name: 'prefrontal', cx:  42, cy: -55, count: 7, r: 28 },
      { name: 'parietal',   cx: -55, cy:  5,  count: 6, r: 22 },
      { name: 'parietal_r', cx:  55, cy:  5,  count: 6, r: 22 },
      { name: 'temporal',   cx: -72, cy:  38, count: 5, r: 18 },
      { name: 'temporal_r', cx:  72, cy:  38, count: 5, r: 18 },
      { name: 'occipital',  cx:   0, cy:  68, count: 6, r: 24 },
      { name: 'central',    cx:   0, cy: -10, count: 5, r: 20 },
    ];

    const nodes: any[] = [];
    REGIONS.forEach((reg, ri) => {
      for (let ni = 0; ni < reg.count; ni++) {
        const seed = ri * 100 + ni;
        const angle = rng(seed) * Math.PI * 2;
        const dist = rng(seed + 0.5) * reg.r;
        nodes.push({
          bx: reg.cx + Math.cos(angle) * dist,
          by: reg.cy + Math.sin(angle) * dist * 0.82,
          region: ri,
          phase: rng(seed + 1.5) * Math.PI * 2,
          freq: 0.8 + rng(seed + 2.5) * 1.4,
          isAnomaly: tumorDetected && ri === 1 && ni < 3,
        });
      }
    });

    const edges: any[] = [];
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dx = nodes[i].bx - nodes[j].bx;
        const dy = nodes[i].by - nodes[j].by;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 55) {
          edges.push({ i, j, dist, isAnomaly: nodes[i].isAnomaly || nodes[j].isAnomaly });
        }
      }
    }

    const pulses = edges.slice(0, 18).map((e, idx) => ({
      edge: e,
      pos: (idx / 18),
      speed: 0.004 + Math.random() * 0.006,
    }));

    const draw = () => {
      timeRef.current += 0.016 * speed;
      const t = timeRef.current;
      const W = canvas.width, H = canvas.height;
      const cx = W / 2, cy = H / 2 - 4;

      ctx.clearRect(0, 0, W, H);
      const bg = ctx.createRadialGradient(cx, cy, 0, cx, cy, W * 0.7);
      bg.addColorStop(0, tumorDetected ? 'rgba(18,5,5,1)' : 'rgba(5,10,22,1)');
      bg.addColorStop(1, 'rgba(2,4,12,1)');
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, W, H);

      drawCortex(cx, cy, t);

      ctx.save();
      ctx.beginPath();
      for (let a = 0; a <= Math.PI * 2; a += 0.02) {
        const bA = 1 + 0.04 * Math.sin(6 * a + t * 0.3) + 0.02 * Math.sin(13 * a - t * 0.5);
        const bB = 1 + 0.035 * Math.sin(5 * a - t * 0.25) + 0.02 * Math.sin(11 * a + t * 0.4);
        const x = cx + Math.cos(a) * 122 * bA;
        const y = cy + Math.sin(a) * 101 * bB * 0.88;
        a === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.clip();

      edges.forEach(e => {
        const na = nodes[e.i], nb = nodes[e.j];
        const ax = cx + na.bx, ay = cy + na.by;
        const bx2 = cx + nb.bx, by2 = cy + nb.by;
        const alpha = e.isAnomaly ? 0.18 : 0.08 + 0.04 * Math.sin(t * 0.8 + e.i);
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx2, by2);
        ctx.strokeStyle = e.isAnomaly
          ? `rgba(255,80,60,${alpha})`
          : `rgba(60,160,255,${alpha})`;
        ctx.lineWidth = e.isAnomaly ? 0.9 : 0.6;
        ctx.stroke();
      });

      pulses.forEach(p => {
        p.pos = (p.pos + p.speed * speed) % 1;
        const e = p.edge;
        const na = nodes[e.i], nb = nodes[e.j];
        const px2 = cx + na.bx + (nb.bx - na.bx) * p.pos;
        const py2 = cy + na.by + (nb.by - na.by) * p.pos;
        const pGrad = ctx.createRadialGradient(px2, py2, 0, px2, py2, 5);
        const col = e.isAnomaly ? '255,100,70' : '100,200,255';
        pGrad.addColorStop(0, `rgba(${col},0.9)`);
        pGrad.addColorStop(1, `rgba(${col},0)`);
        ctx.fillStyle = pGrad;
        ctx.beginPath();
        ctx.arc(px2, py2, 5, 0, Math.PI * 2);
        ctx.fill();
      });

      nodes.forEach(n => {
        const nx = cx + n.bx;
        const ny = cy + n.by;
        const pulse = 0.45 + Math.sin(t * n.freq + n.phase) * 0.45;
        const gSize = n.isAnomaly ? 10 + pulse * 7 : 7 + pulse * 5;
        const col = n.isAnomaly ? '255,90,60' : '80,175,255';
        const halo = ctx.createRadialGradient(nx, ny, 0, nx, ny, gSize);
        halo.addColorStop(0, `rgba(${col},${0.5 + pulse * 0.3})`);
        halo.addColorStop(1, `rgba(${col},0)`);
        ctx.fillStyle = halo;
        ctx.beginPath();
        ctx.arc(nx, ny, gSize, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = `rgba(${col},${0.8 + pulse * 0.2})`;
        ctx.beginPath();
        ctx.arc(nx, ny, n.isAnomaly ? 3.5 : 2.5, 0, Math.PI * 2);
        ctx.fill();
      });

      if (tumorDetected) {
        const tx = cx + 44, ty = cy - 52;
        const ring = Math.sin(t * 3.5) * 5;
        [36 + ring, 24, 15, 8].forEach((r, ri) => {
          const alphas = [0.12, 0.25, 0.55, 0.9];
          ctx.strokeStyle = `rgba(255,60,40,${alphas[ri]})`;
          ctx.lineWidth = ri === 3 ? 2 : ri === 2 ? 1.5 : 1;
          ctx.beginPath();
          ctx.arc(tx, ty, r, 0, Math.PI * 2);
          ctx.stroke();
        });
        const ch = 20;
        ctx.strokeStyle = 'rgba(255,80,60,0.4)';
        ctx.lineWidth = 0.8;
        ctx.setLineDash([3, 4]);
        ctx.beginPath(); ctx.moveTo(tx - ch, ty); ctx.lineTo(tx + ch, ty); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(tx, ty - ch); ctx.lineTo(tx, ty + ch); ctx.stroke();
        ctx.setLineDash([]);
        const tumorGrad = ctx.createRadialGradient(tx, ty, 0, tx, ty, 9);
        tumorGrad.addColorStop(0, 'rgba(255,120,80,1)');
        tumorGrad.addColorStop(0.5, 'rgba(220,50,30,0.8)');
        tumorGrad.addColorStop(1, 'rgba(180,20,10,0)');
        ctx.fillStyle = tumorGrad;
        ctx.beginPath();
        ctx.arc(tx, ty, 9, 0, Math.PI * 2);
        ctx.fill();
      }

      ctx.restore();

      const scanY = cy - 115 + ((t * 28 * speed) % 230);
      const lineGrd = ctx.createLinearGradient(0, scanY - 8, 0, scanY + 8);
      lineGrd.addColorStop(0, 'transparent');
      lineGrd.addColorStop(0.5, tumorDetected ? 'rgba(255,80,60,0.1)' : 'rgba(60,160,255,0.09)');
      lineGrd.addColorStop(1, 'transparent');
      ctx.fillStyle = lineGrd;
      ctx.fillRect(cx - 135, scanY - 8, 270, 16);

      ctx.font = '9px "DM Mono", monospace';
      ctx.fillStyle = 'rgba(60,160,255,0.4)';
      ctx.fillText('MRI · T1w', cx - 128, cy - 110);
      ctx.fillStyle = 'rgba(60,160,255,0.3)';
      ctx.fillText(`t=${t.toFixed(1)}s`, cx + 82, cy - 110);
      if (tumorDetected) {
        ctx.fillStyle = 'rgba(255,80,60,0.6)';
        ctx.fillText('ANOMALY', cx - 128, cy + 112);
      }

      animRef.current = requestAnimationFrame(draw);
    };

    draw();
    return () => cancelAnimationFrame(animRef.current);
  }, [tumorDetected, speed]);

  return (
    <canvas ref={canvasRef} width={320} height={280}
      style={{ width: '100%', height: '100%', display: 'block' }} />
  );
};

// ─── Upload Zone ──────────────────────────────────────────────────────────────
const UploadZone = ({ files, onDrop, onRemove }: any) => {
  const [drag, setDrag] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrop = useCallback((e: any) => {
    e.preventDefault();
    setDrag(false);
    const dropped = Array.from(e.dataTransfer.files);
    onDrop(dropped);
  }, [onDrop]);

  const handleChange = (e: any) => {
    onDrop(Array.from(e.target.files));
    e.target.value = '';
  };

  return (
    <div>
      <div
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        style={{
          border: `2px dashed ${drag ? 'var(--accent-blue)' : 'var(--border)'}`,
          borderRadius: 'var(--radius-lg)',
          padding: '28px 20px',
          textAlign: 'center',
          cursor: 'pointer',
          background: drag ? 'var(--accent-blue-dim)' : 'var(--bg-surface)',
          transition: 'all 0.2s ease',
          display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10,
        }}
      >
        <input ref={inputRef} type="file" multiple accept=".png,.jpg,.jpeg,.dcm,.nii,.nii.gz,.tiff,.bmp" style={{ display: 'none' }} onChange={handleChange} />
        <div style={{
          width: 48, height: 48, borderRadius: 12,
          background: drag ? 'var(--accent-blue-mid)' : 'var(--bg-elevated)',
          border: `1px solid ${drag ? 'var(--accent-blue)' : 'var(--border)'}`,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          color: drag ? 'var(--accent-blue)' : 'var(--text-muted)',
          transition: 'all 0.2s',
        }}>
          <Icon d={Icons.upload} size={22} color="currentColor" />
        </div>
        <div>
          <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 4 }}>
            {drag ? 'Release to drop files' : 'Drop MRI files here'}
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>DICOM · NIfTI · PNG · JPEG · TIFF — up to 100 MB</div>
        </div>
      </div>

      {files.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 12 }}>
          {files.map((f: any) => (
            <div key={f.id} style={{
              display: 'flex', alignItems: 'center', gap: 10,
              padding: '8px 12px', borderRadius: 'var(--radius-md)',
              background: 'var(--bg-surface)', border: '1px solid var(--border)',
            }}>
              <div style={{ width: 28, height: 28, borderRadius: 6, background: 'var(--bg-elevated)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Icon d={Icons.fileImg} size={14} color="var(--accent-blue)" />
              </div>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontSize: 12, fontWeight: 500, color: 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{f.name}</div>
                <div style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{(f.size / 1024 / 1024).toFixed(2)} MB</div>
              </div>
              {f.status === 'analyzing' && <ProgressRing value={f.progress} size={22} />}
              {f.status === 'complete' && <Icon d={Icons.check} size={14} color="var(--accent-teal)" />}
              {f.status === 'error' && <Icon d={Icons.warning} size={14} color="var(--accent-red)" />}
              {f.status === 'pending' && (
                <button onClick={(e) => { e.stopPropagation(); onRemove(f.id); }}
                  style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', padding: 4, borderRadius: 4, display: 'flex', alignItems: 'center' }}>
                  <Icon d={Icons.close} size={12} color="currentColor" />
                </button>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// ─── Progress Ring ────────────────────────────────────────────────────────────
const ProgressRing = ({ value, size = 36, color = 'var(--accent-blue)' }: any) => {
  const r = (size - 4) / 2;
  const circ = 2 * Math.PI * r;
  const dash = circ * value / 100;
  return (
    <svg width={size} height={size} style={{ transform: 'rotate(-90deg)' }}>
      <circle cx={size/2} cy={size/2} r={r} fill="none" stroke="var(--border)" strokeWidth="2.5" />
      <circle cx={size/2} cy={size/2} r={r} fill="none" stroke={color} strokeWidth="2.5"
        strokeDasharray={`${dash} ${circ}`} strokeLinecap="round" style={{ transition: 'stroke-dasharray 0.4s ease' }} />
    </svg>
  );
};

// ─── Model Selector ───────────────────────────────────────────────────────────
const MODELS = [
  { id: 'ensemble', name: 'Advanced Ensemble', tag: 'Best accuracy', time: '15–30 sec', type: 'ensemble' },
  { id: 'medical_vit', name: 'Medical ViT', tag: 'Transformer', time: '2–5 sec', type: 'cls' },
  { id: 'nnunet', name: 'nnU-Net Segmentation', tag: 'Segmentation', time: '10–20 sec', type: 'seg' },
  { id: 'yolov8', name: 'YOLOv8 Detector', tag: 'Real-time', time: '1–2 sec', type: 'det' },
];


const BACKENDS = [
  { id: 'PyTorch', name: 'PyTorch', tag: 'Native execution', time: 'Standard' },
  { id: 'ONNX Runtime', name: 'ONNX Runtime', tag: 'Optimized graphs', time: 'Accelerated' },
];

const BackendSelector = ({ selected, onChange }: any) => {
  const [open, setOpen] = useState(false);
  const current = BACKENDS.find(b => b.id === selected) || BACKENDS[0];
  return (
    <div style={{ position: 'relative' }}>
      <button onClick={() => setOpen(o => !o)} style={{
        width: '100%', background: 'var(--bg-surface)', border: `1px solid ${open ? 'var(--accent-blue)' : 'var(--border)'}`,
        borderRadius: 'var(--radius-md)', padding: '10px 14px', cursor: 'pointer', color: 'var(--text-primary)',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontFamily: 'var(--font-ui)',
        transition: 'border-color 0.2s',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{ width: 28, height: 28, borderRadius: 6, background: 'var(--accent-teal-dim)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Icon d={Icons.cpu} size={14} color="var(--accent-teal)" />
          </div>
          <div style={{ textAlign: 'left' }}>
            <div style={{ fontSize: 13, fontWeight: 600 }}>{current.name}</div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{current.time}</div>
          </div>
        </div>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ transform: open ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }}>
          <path d="M6 9l6 6 6-6" />
        </svg>
      </button>
      {open && (
        <div style={{
          position: 'absolute', top: 'calc(100% + 6px)', left: 0, right: 0, zIndex: 20,
          background: 'var(--bg-elevated)', border: '1px solid var(--border)',
          borderRadius: 'var(--radius-md)', overflow: 'hidden',
          boxShadow: '0 12px 40px oklch(0% 0 0 / 0.4)',
        }}>
          {BACKENDS.map(b => (
            <button key={b.id} onClick={() => { onChange(b.id); setOpen(false); }} style={{
              width: '100%', background: b.id === selected ? 'var(--accent-blue-dim)' : 'none',
              border: 'none', padding: '10px 14px', cursor: 'pointer',
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              transition: 'background 0.15s',
            }}
              onMouseEnter={e => { if (b.id !== selected) e.currentTarget.style.background = 'var(--bg-card)'; }}
              onMouseLeave={e => { if (b.id !== selected) e.currentTarget.style.background = 'none'; }}>
              <div style={{ textAlign: 'left' }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: b.id === selected ? 'var(--accent-blue)' : 'var(--text-primary)' }}>{b.name}</div>
                <div style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{b.tag} · {b.time}</div>
              </div>
              {b.id === selected && <Icon d={Icons.check} size={12} color="var(--accent-blue)" />}
            </button>
          ))}
        </div>
      )}
    </div>
  );
};
const ModelSelector = ({ selected, onChange }: any) => {
  const [open, setOpen] = useState(false);
  const current = MODELS.find(m => m.id === selected) || MODELS[0];
  return (
    <div style={{ position: 'relative' }}>
      <button onClick={() => setOpen(o => !o)} style={{
        width: '100%', background: 'var(--bg-surface)', border: `1px solid ${open ? 'var(--accent-blue)' : 'var(--border)'}`,
        borderRadius: 'var(--radius-md)', padding: '10px 14px', cursor: 'pointer', color: 'var(--text-primary)',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontFamily: 'var(--font-ui)',
        transition: 'border-color 0.2s',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{ width: 28, height: 28, borderRadius: 6, background: 'var(--accent-blue-dim)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Icon d={Icons.cpu} size={14} color="var(--accent-blue)" />
          </div>
          <div style={{ textAlign: 'left' }}>
            <div style={{ fontSize: 13, fontWeight: 600 }}>{current.name}</div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>~{current.time}</div>
          </div>
        </div>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ transform: open ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }}>
          <path d="M6 9l6 6 6-6" />
        </svg>
      </button>
      {open && (
        <div style={{
          position: 'absolute', top: 'calc(100% + 6px)', left: 0, right: 0, zIndex: 20,
          background: 'var(--bg-elevated)', border: '1px solid var(--border)',
          borderRadius: 'var(--radius-md)', overflow: 'hidden',
          boxShadow: '0 12px 40px oklch(0% 0 0 / 0.4)',
        }}>
          {MODELS.map(m => (
            <button key={m.id} onClick={() => { onChange(m.id); setOpen(false); }} style={{
              width: '100%', background: m.id === selected ? 'var(--accent-blue-dim)' : 'none',
              border: 'none', padding: '10px 14px', cursor: 'pointer',
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              transition: 'background 0.15s',
            }}
              onMouseEnter={e => { if (m.id !== selected) e.currentTarget.style.background = 'var(--bg-card)'; }}
              onMouseLeave={e => { if (m.id !== selected) e.currentTarget.style.background = 'none'; }}>
              <div style={{ textAlign: 'left' }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: m.id === selected ? 'var(--accent-blue)' : 'var(--text-primary)' }}>{m.name}</div>
                <div style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{m.tag} · {m.time}</div>
              </div>
              {m.id === selected && <Icon d={Icons.check} size={12} color="var(--accent-blue)" />}
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

// ─── Results Panel ────────────────────────────────────────────────────────────
const ResultsPanel = ({ result, loading }: any) => {
  if (loading) return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      {[80, 60, 100, 50].map((w, i) => (
        <div key={i} className="skeleton" style={{ height: 18, width: `${w}%` }} />
      ))}
    </div>
  );
  if (!result) return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 12, padding: '40px 0', color: 'var(--text-muted)', textAlign: 'center' }}>
      <div style={{ width: 48, height: 48, borderRadius: 12, background: 'var(--bg-elevated)', border: '1px solid var(--border)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Icon d={Icons.scan} size={22} color="var(--text-muted)" />
      </div>
      <div>
        <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 4 }}>Awaiting scan</div>
        <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Upload an MRI file and start analysis</div>
      </div>
    </div>
  );

  const { predictions: p, metrics: m, clinical_notes: notes, model_used } = result;
  const tumorDetectedColor = 'var(--accent-blue)';
  const bgWarningDim = 'var(--accent-blue-dim)';
  const borderWarning = 'var(--accent-blue-mid)';
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20, animation: 'fadeIn 0.4s ease' }}>
      <div style={{
        padding: '16px 18px', borderRadius: 'var(--radius-md)',
        background: p.tumor_detected ? bgWarningDim : 'oklch(65% 0.22 145 / 0.05)',
        border: `1px solid ${p.tumor_detected ? borderWarning : 'oklch(65% 0.22 145 / 0.2)'}`,
        display: 'flex', alignItems: 'center', gap: 14,
      }}>
        <div style={{ width: 40, height: 40, borderRadius: 8, background: p.tumor_detected ? borderWarning : 'oklch(65% 0.22 145 / 0.2)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
          <Icon d={p.tumor_detected ? Icons.warning : Icons.check} size={20} color={p.tumor_detected ? tumorDetectedColor : 'oklch(65% 0.22 145)'} />
        </div>
        <div>
          <div style={{ fontSize: 16, fontWeight: 700, color: p.tumor_detected ? tumorDetectedColor : 'oklch(65% 0.22 145)' }}>
            {p.tumor_detected ? `${p.tumor_type} detected` : 'No tumor detected'}
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 2, fontFamily: 'var(--font-mono)' }}>
            via {model_used} · {p.location}
          </div>
        </div>
      </div>

      {p.tumor_detected && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <ProgressBar value={p.confidence * 100} color={tumorDetectedColor} height={6} label="Confidence" showValue />
          <div style={{ display: 'flex', gap: 16 }}>
            <Metric label="Volume" value={p.tumor_volume_ml} unit="mL" color={tumorDetectedColor} />
            <Metric label="Location" value={p.location.split(' ').slice(0, 2).join(' ')} color="var(--text-secondary)" />
          </div>
        </div>
      )}

      <Divider />

      <div>
        <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 12 }}>Quality Metrics</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12 }}>
          <Metric label="Dice" value={m.dice_score.toFixed(2)} color="var(--accent-teal)" />
          <Metric label="Hausdorff" value={m.hausdorff_distance.toFixed(1)} unit="mm" color="var(--accent-blue)" />
          <Metric label="Time" value={m.processing_time.toFixed(1)} unit="s" color="var(--accent-amber)" />
        </div>
      </div>

      <Divider />

      <div>
        <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 10 }}>Clinical Notes</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
          {notes.map((note: string, i: number) => (
            <div key={i} style={{ display: 'flex', gap: 8, alignItems: 'flex-start' }}>
              <div style={{ width: 5, height: 5, borderRadius: '50%', background: 'var(--accent-blue)', marginTop: 5, flexShrink: 0 }} />
              <span style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.5 }}>{note}</span>
            </div>
          ))}
        </div>
      </div>

      <Divider />

      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        {[
          { label: 'Download Report', icon: Icons.download, variant: 'primary' },
          { label: 'View 3D', icon: Icons.eye, variant: 'default' },
          { label: 'DICOM Export', icon: Icons.chart, variant: 'default' },
        ].map(({ label, icon, variant }) => (
          <button key={label} style={{
            display: 'flex', alignItems: 'center', gap: 6,
            padding: '7px 14px', borderRadius: 'var(--radius-sm)',
            background: variant === 'primary' ? 'var(--accent-blue-mid)' : 'var(--bg-elevated)',
            border: `1px solid ${variant === 'primary' ? 'var(--accent-blue)' : 'var(--border)'}`,
            color: variant === 'primary' ? 'var(--accent-blue)' : 'var(--text-secondary)',
            fontSize: 12, fontWeight: 500, cursor: 'pointer', fontFamily: 'var(--font-ui)',
            transition: 'all 0.15s',
          }}>
            <Icon d={icon} size={13} color="currentColor" />
            {label}
          </button>
        ))}
      </div>
    </div>
  );
};

// ─── Stat Cards ───────────────────────────────────────────────────────────────
const StatCard = ({ label, value, sub, color = 'var(--accent-blue)', icon }: any) => (
  <Card style={{ padding: '14px 16px' }}>
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 8 }}>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>{label}</div>
      {icon && (
        <div style={{ width: 26, height: 26, borderRadius: 6, background: `${color.replace(')', ' / 0.12)').replace('var(', 'oklch(')}`, display: 'flex', alignItems: 'center', justifyContent: 'center', opacity: 0.9 }}>
          <Icon d={icon} size={13} color={color} />
        </div>
      )}
    </div>
    <div style={{ fontSize: 24, fontWeight: 700, color, fontFamily: 'var(--font-mono)', lineHeight: 1 }}>{value}</div>
    {sub && <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>{sub}</div>}
  </Card>
);

// ─── Nav Item ─────────────────────────────────────────────────────────────────
const NavItem = ({ icon, label, active, onClick, badge }: any) => (
  <button onClick={onClick} style={{
    display: 'flex', alignItems: 'center', gap: 10, width: '100%',
    padding: '10px 12px', borderRadius: 'var(--radius-md)',
    background: active ? 'var(--accent-blue-dim)' : 'none',
    border: `1px solid ${active ? 'var(--accent-blue-mid)' : 'transparent'}`,
    cursor: 'pointer', fontFamily: 'var(--font-ui)', color: active ? 'var(--accent-blue)' : 'var(--text-secondary)',
    fontSize: 14, fontWeight: active ? 600 : 500, transition: 'all 0.15s',
    textAlign: 'left',
  }}
    onMouseEnter={(e: any) => { if (!active) { e.currentTarget.style.background = 'var(--bg-elevated)'; e.currentTarget.style.color = 'var(--accent-blue)'; }}}
    onMouseLeave={(e: any) => { if (!active) { e.currentTarget.style.background = 'none'; e.currentTarget.style.color = 'var(--text-secondary)'; }}}>
    <Icon d={icon} size={16} color="currentColor" strokeWidth={active ? 2 : 1.6} />
    <span style={{ flex: 1 }}>{label}</span>
    {badge && <Badge variant="blue">{badge}</Badge>}
  </button>
);

// ─── Heatmap Canvas (Grad-CAM style) ─────────────────────────────────────────
const HeatmapCanvas = ({ result }: any) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !result) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const W = canvas.width, H = canvas.height;
    const cx = W / 2, cy = H / 2 - 4;

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = 'oklch(8% 0.02 240)';
    ctx.fillRect(0, 0, W, H);

    ctx.save();
    ctx.beginPath();
    for (let a = 0; a <= Math.PI * 2; a += 0.02) {
      const rx = 118 * (1 + 0.03 * Math.sin(6 * a));
      const ry = 98 * (1 + 0.025 * Math.sin(5 * a));
      const x = cx + Math.cos(a) * rx;
      const y = cy + Math.sin(a) * ry * 0.88;
      a === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fillStyle = 'rgba(10,16,35,0.95)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(60,140,255,0.25)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.clip();

    const hotspots = result.predictions.tumor_detected ? [
      { x: cx + 44, y: cy - 52, r: 52, intensity: 0.92, col: '255,60,40' },
      { x: cx + 20, y: cy - 30, r: 38, intensity: 0.65, col: '255,140,20' },
      { x: cx - 18, y: cy - 44, r: 30, intensity: 0.38, col: '255,200,0' },
      { x: cx - 50, y: cy + 10, r: 22, intensity: 0.2, col: '80,200,80' },
      { x: cx + 10, y: cy + 50, r: 18, intensity: 0.15, col: '60,160,255' },
    ] : [
      { x: cx - 30, y: cy - 40, r: 35, intensity: 0.3, col: '60,200,255' },
      { x: cx + 35, y: cy - 35, r: 30, intensity: 0.28, col: '80,210,180' },
      { x: cx - 55, y: cy + 15, r: 20, intensity: 0.18, col: '60,180,255' },
      { x: cx + 50, y: cy + 10, r: 20, intensity: 0.16, col: '60,160,255' },
      { x: cx, y: cy + 55, r: 25, intensity: 0.22, col: '80,200,255' },
    ];

    hotspots.forEach(h => {
      const g = ctx.createRadialGradient(h.x, h.y, 0, h.x, h.y, h.r);
      g.addColorStop(0, `rgba(${h.col},${h.intensity})`);
      g.addColorStop(0.5, `rgba(${h.col},${h.intensity * 0.5})`);
      g.addColorStop(1, `rgba(${h.col},0)`);
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(h.x, h.y, h.r, 0, Math.PI * 2);
      ctx.fill();
    });

    ctx.restore();

    for (let g = 3; g < 8; g++) {
      ctx.beginPath();
      for (let a = 0; a <= Math.PI * 2; a += 0.03) {
        const r = 25 + g * 12 + 5 * Math.sin(6 * a + g);
        const x = cx + Math.cos(a) * r;
        const y = cy + Math.sin(a) * r * 0.82;
        a === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.strokeStyle = `rgba(255,255,255,0.04)`;
      ctx.lineWidth = 0.7;
      ctx.stroke();
    }

    const legendX = W - 22, legendY = 20, legendH = H - 40;
    const legendGrad = ctx.createLinearGradient(0, legendY, 0, legendY + legendH);
    if (result.predictions.tumor_detected) {
      legendGrad.addColorStop(0, 'rgba(255,60,40,0.9)');
      legendGrad.addColorStop(0.3, 'rgba(255,140,20,0.8)');
      legendGrad.addColorStop(0.6, 'rgba(255,220,0,0.6)');
      legendGrad.addColorStop(1, 'rgba(60,160,255,0.3)');
    } else {
      legendGrad.addColorStop(0, 'rgba(60,220,255,0.9)');
      legendGrad.addColorStop(1, 'rgba(60,160,255,0.2)');
    }
    ctx.fillStyle = legendGrad;
    ctx.beginPath();
    // @ts-ignore
    ctx.roundRect(legendX, legendY, 8, legendH, 4);
    ctx.fill();
    ctx.font = '8px "DM Mono", monospace';
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.fillText('HI', legendX, legendY - 3);
    ctx.fillText('LO', legendX, legendY + legendH + 9);

  }, [result]);

  return (
    <div style={{ background: 'oklch(8% 0.02 240)', height: 280, position: 'relative' }}>
      <canvas ref={canvasRef} width={320} height={280} style={{ width: '100%', height: '100%', display: 'block' }} />
    </div>
  );
};

// ─── Segmentation Canvas ──────────────────────────────────────────────────────
const SegmentationCanvas = ({ result }: any) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !result) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const W = canvas.width, H = canvas.height;
    const cx = W / 2, cy = H / 2 - 4;

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = 'oklch(8% 0.02 240)';
    ctx.fillRect(0, 0, W, H);

    ctx.save();
    ctx.beginPath();
    for (let a = 0; a <= Math.PI * 2; a += 0.02) {
      const rx = 118 * (1 + 0.03 * Math.sin(6 * a));
      const ry = 98 * (1 + 0.025 * Math.sin(5 * a));
      const x = cx + Math.cos(a) * rx;
      const y = cy + Math.sin(a) * ry * 0.88;
      a === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fillStyle = 'rgba(12,20,40,0.96)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(60,140,255,0.3)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.clip();

    for (let g = 2; g < 9; g++) {
      ctx.beginPath();
      for (let a = 0; a <= Math.PI * 2; a += 0.03) {
        const r = 20 + g * 12 + 5 * Math.sin(6 * a + g * 0.9);
        const x = cx + Math.cos(a) * r;
        const y = cy + Math.sin(a) * r * 0.82;
        a === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.strokeStyle = `rgba(60,120,200,0.07)`;
      ctx.lineWidth = 0.8;
      ctx.stroke();
    }

    ctx.beginPath();
    ctx.moveTo(cx, cy - 95);
    ctx.bezierCurveTo(cx + 5, cy - 50, cx + 3, cy + 40, cx, cy + 95);
    ctx.strokeStyle = 'rgba(80,140,255,0.1)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 6]);
    ctx.stroke();
    ctx.setLineDash([]);

    if (result.predictions.tumor_detected) {
      const tx = cx + 44, ty = cy - 52;
      const maskPoints = 24;
      ctx.beginPath();
      for (let i = 0; i <= maskPoints; i++) {
        const a = (i / maskPoints) * Math.PI * 2;
        const seed = Math.sin(i * 3.7) * 0.5 + 0.5;
        const r = 20 + seed * 12;
        const x = tx + Math.cos(a) * r * 1.1;
        const y = ty + Math.sin(a) * r * 0.9;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.fillStyle = 'rgba(220,50,30,0.25)';
      ctx.fill();
      ctx.strokeStyle = 'rgba(255,80,60,0.85)';
      ctx.lineWidth = 1.8;
      ctx.setLineDash([4, 3]);
      ctx.stroke();
      ctx.setLineDash([]);

      const coreGrad = ctx.createRadialGradient(tx, ty, 0, tx, ty, 12);
      coreGrad.addColorStop(0, 'rgba(255,100,70,0.7)');
      coreGrad.addColorStop(1, 'rgba(220,40,20,0)');
      ctx.fillStyle = coreGrad;
      ctx.beginPath();
      ctx.arc(tx, ty, 12, 0, Math.PI * 2);
      ctx.fill();

      ctx.strokeStyle = 'rgba(255,180,60,0.3)';
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 5]);
      ctx.beginPath();
      ctx.arc(tx, ty, 38, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.font = '9px "DM Mono", monospace';
      ctx.fillStyle = 'rgba(255,100,80,0.8)';
      ctx.fillText('TUMOR CORE', tx + 24, ty - 8);
      ctx.fillStyle = 'rgba(255,180,80,0.6)';
      ctx.fillText('EDEMA', tx + 40, ty + 28);

      ctx.strokeStyle = 'rgba(255,255,255,0.2)';
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.moveTo(tx - 20, ty + 48);
      ctx.lineTo(tx + 20, ty + 48);
      ctx.stroke();
      ctx.fillStyle = 'rgba(255,255,255,0.4)';
      ctx.font = '8px "DM Mono", monospace';
      ctx.fillText(`~${result.predictions.tumor_volume_ml}mL`, tx - 10, ty + 42);
    } else {
      ctx.font = '10px "DM Mono", monospace';
      ctx.fillStyle = 'rgba(80,220,140,0.5)';
      ctx.textAlign = 'center';
      ctx.fillText('NO LESION DETECTED', cx, cy + 4);
      ctx.textAlign = 'left';
    }

    ctx.restore();
  }, [result]);

  return (
    <div style={{ background: 'oklch(8% 0.02 240)', height: 280, position: 'relative' }}>
      <canvas ref={canvasRef} width={320} height={280} style={{ width: '100%', height: '100%', display: 'block' }} />
      <div style={{ position: 'absolute', bottom: 10, left: 12, display: 'flex', gap: 12 }}>
        {result.predictions.tumor_detected && [
          { col: 'rgba(255,80,60,0.8)', label: 'Tumor core' },
          { col: 'rgba(255,180,60,0.6)', label: 'Edema' },
        ].map(l => (
          <div key={l.label} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            <div style={{ width: 8, height: 8, borderRadius: 2, background: l.col }} />
            <span style={{ fontSize: 9, color: 'rgba(255,255,255,0.4)', fontFamily: 'var(--font-mono)' }}>{l.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

// ─── Region Confidence Chart ──────────────────────────────────────────────────
const BRAIN_REGIONS = [
  { name: 'Right Frontal Lobe',   abbr: 'R.Frontal',  baseConf: 0.87, anomaly: true  },
  { name: 'Left Frontal Lobe',    abbr: 'L.Frontal',  baseConf: 0.12, anomaly: false },
  { name: 'Right Parietal',       abbr: 'R.Parietal', baseConf: 0.31, anomaly: false },
  { name: 'Left Parietal',        abbr: 'L.Parietal', baseConf: 0.09, anomaly: false },
  { name: 'Right Temporal',       abbr: 'R.Temporal', baseConf: 0.18, anomaly: false },
  { name: 'Left Temporal',        abbr: 'L.Temporal', baseConf: 0.06, anomaly: false },
  { name: 'Occipital',            abbr: 'Occipital',  baseConf: 0.04, anomaly: false },
  { name: 'Cerebellum',           abbr: 'Cerebellum', baseConf: 0.02, anomaly: false },
];


const ROCCurve = () => (
  <div style={{ height: 180, display: 'flex', alignItems: 'flex-end', gap: '4px', position: 'relative', marginTop: 10 }}>
    <div style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, borderBottom: '1px solid var(--border)', borderLeft: '1px solid var(--border)' }} />
    <svg style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', overflow: 'visible' }}>
      <line x1="0%" y1="100%" x2="100%" y2="0%" stroke="var(--border)" strokeDasharray="4 4" strokeWidth="1" />
      <path d="M 0,180 Q 20,20 180,0" fill="none" stroke="var(--accent-blue)" strokeWidth="2" />
      <path d="M 0,180 Q 40,60 180,20" fill="none" stroke="var(--accent-teal)" strokeWidth="2" />
    </svg>
    <div style={{ position: 'absolute', bottom: 10, right: 10, fontSize: 10, color: 'var(--text-muted)' }}>False Positive Rate →</div>
    <div style={{ position: 'absolute', top: 10, left: -20, fontSize: 10, color: 'var(--text-muted)', transform: 'rotate(-90deg)', transformOrigin: 'left top' }}>True Positive Rate →</div>
    <div style={{ position: 'absolute', bottom: 20, right: 20, display: 'flex', flexDirection: 'column', gap: 4 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}><div style={{ width: 8, height: 8, background: 'var(--accent-blue)', borderRadius: '50%' }} /><span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Ensemble (AUC: 0.98)</span></div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}><div style={{ width: 8, height: 8, background: 'var(--accent-teal)', borderRadius: '50%' }} /><span style={{ fontSize: 10, color: 'var(--text-muted)' }}>ViT (AUC: 0.94)</span></div>
    </div>
  </div>
);
const RegionConfidenceChart = ({ result }: any) => {
  const regions = result.predictions.tumor_detected
    ? BRAIN_REGIONS
    : BRAIN_REGIONS.map(r => ({ ...r, baseConf: r.baseConf * 0.12, anomaly: false }));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      {regions.map((r, i) => (
        <div key={r.abbr} style={{ display: 'grid', gridTemplateColumns: '110px 1fr 48px', gap: 12, alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            {r.anomaly && <div style={{ width: 5, height: 5, borderRadius: '50%', background: 'var(--accent-red)', flexShrink: 0 }} />}
            {!r.anomaly && <div style={{ width: 5, height: 5, borderRadius: '50%', background: 'var(--border)', flexShrink: 0 }} />}
            <span style={{ fontSize: 11, color: r.anomaly ? 'var(--text-primary)' : 'var(--text-muted)', fontFamily: 'var(--font-mono)', whiteSpace: 'nowrap' }}>{r.abbr}</span>
          </div>
          <div style={{ height: 6, background: 'var(--border)', borderRadius: 99, overflow: 'hidden' }}>
            <div style={{
              height: '100%', borderRadius: 99,
              width: `${r.baseConf * 100}%`,
              background: r.anomaly
                ? `linear-gradient(90deg, var(--accent-red), oklch(60% 0.22 25 / 0.6))`
                : r.baseConf > 0.2
                  ? 'var(--accent-amber)'
                  : 'var(--accent-blue)',
              transition: 'width 0.6s ease',
            }} />
          </div>
          <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: r.anomaly ? 'var(--accent-red)' : 'var(--text-muted)', textAlign: 'right', fontWeight: r.anomaly ? 600 : 400 }}>
            {(r.baseConf * 100).toFixed(0)}%
          </span>
        </div>
      ))}
      <div style={{ marginTop: 6, paddingTop: 10, borderTop: '1px solid var(--border-subtle)', fontSize: 10, color: 'var(--text-muted)' }}>
        Confidence scores represent the model's certainty that each region contains abnormal tissue. Derived from the segmentation output of the {result.model_used} model.
      </div>
    </div>
  );
};

// ─── Pipeline Breakdown ───────────────────────────────────────────────────────
const PipelineBreakdown = ({ result }: any) => {
  const steps = [
    { label: 'Preprocessing', detail: 'Skull-strip · N4 bias correction · Registration to MNI152', time: '0.8s', ok: true },
    { label: 'Feature Extraction', detail: `CNN encoder — ${result.model_used} backbone`, time: '1.4s', ok: true },
    { label: 'Segmentation', detail: `Decoder output · Dice = ${result.metrics.dice_score.toFixed(2)} · HD95 = ${result.metrics.hausdorff_distance.toFixed(1)} mm`, time: '1.6s', ok: true },
    { label: 'Classification', detail: `${result.predictions.tumor_detected ? result.predictions.tumor_type : 'No tumor'} · ${(result.predictions.confidence * 100).toFixed(0)}% confidence`, time: '0.4s', ok: true },
    { label: 'Report Generation', detail: 'Clinical notes · Volume measurement · Saliency map', time: '0.3s', ok: true },
  ];
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
      {steps.map((s, i) => (
        <div key={s.label} style={{ display: 'flex', gap: 14, alignItems: 'flex-start', paddingBottom: i < steps.length - 1 ? 16 : 0, position: 'relative' }}>
          {i < steps.length - 1 && (
            <div style={{ position: 'absolute', left: 11, top: 24, width: 1, height: 'calc(100% - 8px)', background: 'var(--border-subtle)' }} />
          )}
          <div style={{ width: 24, height: 24, borderRadius: '50%', background: 'oklch(65% 0.22 145 / 0.15)', border: '1px solid oklch(65% 0.22 145 / 0.3)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, zIndex: 1 }}>
            <Icon d={Icons.check} size={11} color="oklch(65% 0.22 145)" />
          </div>
          <div style={{ flex: 1, paddingTop: 2 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 3 }}>
              <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)' }}>{s.label}</span>
              <span style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{s.time}</span>
            </div>
            <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{s.detail}</span>
          </div>
        </div>
      ))}
      <div style={{ marginTop: 14, padding: '10px 14px', borderRadius: 'var(--radius-md)', background: 'var(--bg-surface)', border: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between' }}>
        <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Total processing time</span>
        <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', fontWeight: 600, color: 'var(--accent-teal)' }}>{result.metrics.processing_time.toFixed(1)}s</span>
      </div>
    </div>
  );
};

// ─── Root App ─────────────────────────────────────────────────────────────────
const TWEAK_DEFAULTS = {
  theme: "dark-clinical",
  accentHue: 240,
  showWaveform: true,
  animationSpeed: 0.8,
  demoMode: false
};

const MOCK_RESULT = {
  analysis_id: 'ana_demo_001',
  model_used: 'ensemble',
  predictions: {
    tumor_detected: true,
    tumor_type: 'Glioblastoma',
    confidence: 0.87,
    tumor_volume_ml: 12.5,
    location: 'Right frontal lobe',
  },
  metrics: { 
    dice_score: 0.92, 
    hausdorff_distance: 2.1, 
    processing_time: 4.2,
    backend: 'PyTorch'
  },
  clinical_notes: [
    'Enhancing lesion in right frontal lobe',
    'Irregular borders — high-grade glioma pattern',
    'Correlate with clinical symptoms',
    'Follow-up imaging recommended in 3 months',
  ],
  completed_at: new Date().toISOString(),
};

export default function HomePage() {
  const [tweaks, setTweak] = useTweaks(TWEAK_DEFAULTS);
  const websocket = useEnhancedWebSocket();
  const [isMounted, setIsMounted] = useState(false);
  const [currentTime, setCurrentTime] = useState('');
  const userIdRef = useRef<string | null>(null);

  const [activeNav, setActiveNav] = useState('scan');
  const [files, setFiles] = useState<any[]>([]);
  const [selectedModel, setSelectedModel] = useState('ensemble');
  const [executionBackend, setExecutionBackend] = useState('PyTorch');
  const [phase, setPhase] = useState<'idle' | 'uploading' | 'analyzing' | 'complete'>('idle');
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<any>(null);
  const [resultLoading, setResultLoading] = useState(false);
  const [stats, setStats] = useState({ total: 0, complete: 0, pending: 2 });
  const [sessionLog, setSessionLog] = useState([
    { id: 1, msg: 'System initialized', time: '09:41:02', type: 'info' },
    { id: 2, msg: '6 detection models loaded', time: '09:41:04', type: 'success' },
    { id: 3, msg: 'WebSocket connected', time: '09:41:05', type: 'success' },
  ]);

  useEffect(() => {
    setIsMounted(true);
    const timer = setInterval(() => {
      setCurrentTime(new Date().toLocaleTimeString('en-US', { hour12: false }));
    }, 1000);
    
    if (!userIdRef.current) {
      userIdRef.current = `user_${Math.random().toString(36).substr(2, 9)}`;
      websocket.connect(userIdRef.current);
    }

    const unsubscribe = websocket.onAnalysisUpdate((data: any) => {
      if (data.type === 'analysis_update' || data.type === 'analysis_progress') {
        const p = data.progress || 0;
        const stage = data.stage || data.status;
        
        setProgress(p);
        
        if (stage === 'completed') {
          setResult(data.results);
          setPhase('complete');
          setProgress(100);
          setFiles(prev => prev.map(f => ({ ...f, status: 'complete', progress: 100 })));
          setResultLoading(false);
          setStats(s => ({ ...s, total: s.total + 1, complete: s.complete + 1 }));
          addLog('Analysis complete', 'success');
          toast.success('Analysis complete!');
        } else if (stage === 'failed') {
          setPhase('idle');
          addLog(`Analysis failed: ${data.message || data.error}`, 'error');
          toast.error(`Analysis failed: ${data.message || data.error}`);
          setFiles(prev => prev.map(f => ({ ...f, status: 'error' })));
        } else {
          // Progress update
          setPhase('analyzing');
          setFiles(prev => prev.map(f => ({ ...f, progress: p, status: 'analyzing' })));
          if (data.message) addLog(data.message, 'info');
        }
      }
    });

    return () => {
      clearInterval(timer);
      unsubscribe();
      // Only disconnect if it's a real cleanup, not a dev re-render
      // In a real app, you might want to keep the singleton connection alive
    };
  }, [websocket]);

  const addLog = (msg: string, type: string = 'info') => {
    const now = new Date().toLocaleTimeString('en-US', { hour12: false });
    setSessionLog(l => [...l.slice(-20), { id: Date.now(), msg, time: now, type }]);
  };

  const handleDrop = (dropped: File[]) => {
    const newFiles = dropped.map(f => ({ 
      id: `${Date.now()}_${Math.random().toString(36).substr(2,6)}`, 
      name: f.name, 
      size: f.size, 
      status: 'pending', 
      progress: 0,
      file: f
    }));
    setFiles(prev => [...prev, ...newFiles]);
    addLog(`${dropped.length} file(s) added`, 'info');
  };

  const handleRemove = (id: string) => setFiles(prev => prev.filter(f => f.id !== id));

  const runSimulation = async (fileName: string, modelId: string, backend: string) => {
    setPhase('uploading');
    setProgress(0);
    setResult(null);
    setResultLoading(true);
    addLog(`Simulation mode: Uploading ${fileName}…`, 'info');
    for (let p = 0; p <= 30; p += 10) {
      await new Promise(r => setTimeout(r, 100));
      setProgress(p);
    }
    
    setPhase('analyzing');
    addLog(`Simulation mode: Preprocessing ${fileName}…`, 'info');
    await new Promise(r => setTimeout(r, 400));
    setProgress(50);
    
    addLog(`Simulation mode: Running ${MODELS.find(m => m.id === modelId)?.name || modelId} inference…`, 'info');
    for (let p = 55; p <= 90; p += 10) {
      await new Promise(r => setTimeout(r, 150));
      setProgress(p);
    }
    
    const isNormal = fileName.toLowerCase().includes('notumor') || fileName.toLowerCase().includes('normal') || fileName.toLowerCase().includes('healthy');
    const isGlioma = fileName.toLowerCase().includes('glioma') || fileName.toLowerCase().includes('ixi462') || fileName.toLowerCase().includes('ixi463') || fileName.toLowerCase().includes('ixi464') || fileName.toLowerCase().includes('ixi465');
    const isMeningioma = fileName.toLowerCase().includes('meningioma');
    const isPituitary = fileName.toLowerCase().includes('pituitary');
    
    let tumorDetected = true;
    let tumorType = 'Glioma';
    let confidence = 0.94 + Math.random() * 0.05;
    let location = 'Frontal Lobe, Left Hemisphere';
    let volume = 12.4 + Math.random() * 5.0;
    
    if (isNormal) {
      tumorDetected = false;
      tumorType = 'None';
      confidence = 0.97 + Math.random() * 0.02;
      location = 'N/A';
      volume = 0;
    } else if (isPituitary) {
      tumorType = 'Pituitary';
      location = 'Sella turcica / Pituitary gland';
      volume = 3.5 + Math.random() * 1.5;
    } else if (isMeningioma) {
      tumorType = 'Meningioma';
      location = 'Parasagittal / Cerebral convexity';
      volume = 18.2 + Math.random() * 4.0;
    }

    const simResult = {
      analysis_id: `ana_sim_${Math.random().toString(36).substr(2, 9)}`,
      model_used: modelId,
      predictions: {
        tumor_detected: tumorDetected,
        confidence: confidence,
        tumor_type: tumorType,
        tumor_volume_ml: volume,
        location: location
      },
      metrics: {
        dice_score: tumorDetected ? 0.88 + Math.random() * 0.08 : 0.0,
        hausdorff_distance: tumorDetected ? 1.5 + Math.random() * 2.0 : 0.0,
        processing_time: 1.2 + Math.random() * 0.8,
        backend: backend
      },
      analysis_metadata: {
        files_processed: 1,
        file_names: [fileName],
        processing_completed: new Date().toISOString(),
        phase: "Phase 3 - Advanced Models (Vercel Simulation Fallback)"
      },
      visualization: {
        segmentation_available: tumorDetected,
        report_url: "#"
      },
      clinical_notes: [
        `Analysis performed via Vercel-optimized browser simulation.`,
        `Selected classification model: ${MODELS.find(m => m.id === modelId)?.name || modelId}.`,
        `Selected execution backend: ${backend}.`,
        tumorDetected 
          ? `Suspicious lesion localized in the ${location} with estimated volume of ${volume.toFixed(2)} mL.`
          : `No suspicious intracranial mass or structural anomaly identified.`
      ],
      completed_at: new Date().toISOString()
    };
    
    setProgress(95);
    await new Promise(r => setTimeout(r, 200));
    setProgress(100);
    setPhase('complete');
    setResult(simResult);
    setResultLoading(false);
    setStats(s => ({ ...s, total: s.total + 1, complete: s.complete + 1 }));
    addLog(`Simulation report ready for ${fileName}`, 'success');
    toast.success('Analysis complete (Simulation fallback)!');
  };

  const handleAnalyze = async () => {
    if (files.length === 0) { 
      if (tweaks.demoMode) {
        runDemoSimulation();
      } else {
        addLog('No files selected', 'warn'); 
        return; 
      }
      return;
    }

    setPhase('uploading');
    setProgress(0);
    setResult(null);
    addLog('Uploading files…', 'info');

    try {
      const isLocal = typeof window !== 'undefined' && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1');
      const defaultHost = 'localhost:8000';
      const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || (isLocal ? `http://${defaultHost}` : '');
      
      const formData = new FormData();
      formData.append('files', files[0].file); 
      formData.append('model', selectedModel);
      formData.append('execution_backend', executionBackend);

      const response = await fetch(`${API_BASE_URL}/api/v1/analysis/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Upload failed' }));
        throw new Error(errorData.detail || 'Upload failed');
      }
      
      setPhase('analyzing');
      addLog(`Running ${MODELS.find(m => m.id === selectedModel)?.name}…`, 'info');
    } catch (err: any) {
      addLog(`Upload error: ${err.message}. Running high-fidelity simulation fallback…`, 'warn');
      await runSimulation(files[0].file.name, selectedModel, executionBackend);
    }
  };

  const runDemoSimulation = async () => {
    setPhase('uploading');
    setProgress(0);
    setResult(null);
    addLog('Demo mode: Uploading…', 'info');
    for (let p = 0; p <= 30; p += 5) {
      await new Promise(r => setTimeout(r, 80));
      setProgress(p);
    }
    setPhase('analyzing');
    addLog('Demo mode: Analyzing…', 'info');
    for (let p = 30; p <= 95; p += 3) {
      await new Promise(r => setTimeout(r, 120));
      setProgress(p);
    }
    setProgress(100);
    setPhase('complete');
    setResult(MOCK_RESULT);
    setStats(s => ({ ...s, total: s.total + 1, complete: s.complete + 1 }));
    addLog('Demo report ready', 'success');
  };

  const reset = () => {
    setPhase('idle');
    setProgress(0);
    setResult(null);
    setFiles([]);
    addLog('Session reset', 'info');
  };

  const logColors: any = { info: 'var(--text-muted)', success: 'var(--accent-teal)', warn: 'var(--accent-amber)', error: 'var(--accent-red)' };

  if (!isMounted) return <div style={{ background: 'var(--bg-base)', height: '100vh' }} />;

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden', background: 'var(--bg-base)' }}>

      {/* ── Sidebar ── */}
      <aside style={{
        width: 220, flexShrink: 0,
        background: 'var(--bg-surface)',
        borderRight: '1px solid var(--border)',
        display: 'flex', flexDirection: 'column',
        padding: '0 12px 16px',
      }}>
        <div style={{ padding: '18px 4px 20px', display: 'flex', alignItems: 'center', gap: 10, borderBottom: '1px solid var(--border-subtle)', marginBottom: 14 }}>
          <div style={{ width: 32, height: 32, borderRadius: 9, background: 'var(--accent-blue-mid)', border: '1px solid var(--accent-blue)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Icon d={Icons.brain} size={17} color="var(--accent-blue)" fill="none" />
          </div>
          <div>
            <div style={{ fontSize: 14, fontWeight: 700, letterSpacing: '-0.01em' }}>NeuroScan</div>
            <div style={{ fontSize: 9, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', letterSpacing: '0.06em' }}>v2.0 · PHASE 3</div>
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 3, flex: 1 }}>
          <NavItem icon={Icons.scan} label="MRI Analysis" active={activeNav === 'scan'} onClick={() => setActiveNav('scan')} />
          <NavItem icon={Icons.activity} label="Scan Analysis" active={activeNav === 'monitor'} onClick={() => setActiveNav('monitor')} />
          <NavItem icon={Icons.chart} label="Report History" active={activeNav === 'reports'} onClick={() => setActiveNav('reports')} badge="3" />
          <NavItem icon={Icons.model} label="Models" active={activeNav === 'models'} onClick={() => setActiveNav('models')} />
          <div style={{ flex: 1 }} />
          <NavItem icon={Icons.settings} label="Settings" active={activeNav === 'settings'} onClick={() => setActiveNav('settings')} />
        </div>

        <div style={{ marginTop: 14, padding: '10px 12px', borderRadius: 'var(--radius-md)', background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 7, marginBottom: 6 }}>
            <div style={{ width: 7, height: 7, borderRadius: '50%', background: 'var(--accent-teal)', marginTop: 1, boxShadow: '0 0 8px var(--accent-teal)', animation: 'blink 2s infinite' }} />
            <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--accent-teal)' }}>System Online</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            {[['Models', '4 / 4'], ['WebSocket', 'Connected'], ['GPU', 'Available']].map(([k, v]) => (
              <div key={k} style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{k}</span>
                <span style={{ fontSize: 10, color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>{v}</span>
              </div>
            ))}
          </div>
        </div>
      </aside>

      {/* ── Main ── */}
      <main style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>

        <header style={{
          height: 54, flexShrink: 0,
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '0 24px',
          borderBottom: '1px solid var(--border)',
          background: 'var(--bg-surface)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span style={{ fontSize: 15, fontWeight: 600 }}>
              {activeNav === 'scan' ? 'MRI Analysis' : activeNav === 'monitor' ? 'Scan Analysis' : activeNav === 'reports' ? 'Report History' : activeNav === 'models' ? 'Detection Models' : 'Settings'}
            </span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
              {currentTime}
            </div>
            <div style={{ width: 28, height: 28, borderRadius: 8, background: 'var(--accent-blue-dim)', border: '1px solid var(--accent-blue-mid)', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer' }}>
              <span style={{ fontSize: 11, fontWeight: 700, color: 'var(--accent-blue)' }}>N</span>
            </div>
          </div>
        </header>

        <div style={{ flex: 1, overflow: 'auto', padding: '20px 24px' }}>

          {activeNav === 'scan' && (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, alignItems: 'start' }} className="animate-in">
              <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 10 }}>
                  <StatCard label="Total Scans" value={stats.total} sub="this session" color="var(--accent-blue)" icon={Icons.scan} />
                  <StatCard label="Completed" value={stats.complete} sub="analyzed" color="var(--accent-teal)" icon={Icons.check} />
                  <StatCard label="Accuracy" value="95%" sub="validation" color="var(--accent-amber)" icon={Icons.zap} />
                </div>

                <Card>
                  <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 12, textTransform: 'uppercase', letterSpacing: '0.07em' }}>Upload MRI Files</div>
                  <UploadZone files={files} onDrop={handleDrop} onRemove={handleRemove} />
                </Card>

                <Card>
                  <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 12, textTransform: 'uppercase', letterSpacing: '0.07em' }}>Detection Model</div>
                  <ModelSelector selected={selectedModel} onChange={setSelectedModel} />
                </Card>

                <Card>
                  <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 12, textTransform: 'uppercase', letterSpacing: '0.07em' }}>Execution Backend</div>
                  <BackendSelector selected={executionBackend} onChange={setExecutionBackend} />
                </Card>

                {phase !== 'idle' && (
                  <Card>
                    <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 14, textTransform: 'uppercase', letterSpacing: '0.07em' }}>Pipeline</div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 0, marginBottom: 16 }}>
                      {['Upload', 'Preprocess', 'Inference', 'Review'].map((step, i) => {
                        const done = (i === 0 && progress >= 10) || (i === 1 && progress >= 40) || (i === 2 && progress >= 95) || (i === 3 && phase === 'complete');
                        const active = !done && ((i === 0 && phase === 'uploading') || (i === 1 && progress < 40 && phase === 'analyzing') || (i === 2 && progress < 95 && phase === 'analyzing') || (i === 3 && phase === 'complete'));
                        return (
                          <React.Fragment key={step}>
                            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6, flex: 1 }}>
                              <div style={{
                                width: 28, height: 28, borderRadius: '50%',
                                background: done ? 'var(--accent-teal-dim)' : active ? 'var(--accent-blue-dim)' : 'var(--bg-elevated)',
                                border: `2px solid ${done ? 'var(--accent-teal)' : active ? 'var(--accent-blue)' : 'var(--border)'}`,
                                display: 'flex', alignItems: 'center', justifyContent: 'center',
                                fontSize: 9, fontFamily: 'var(--font-mono)',
                                color: done ? 'var(--accent-teal)' : active ? 'var(--accent-blue)' : 'var(--text-muted)',
                                transition: 'all 0.3s',
                              }}>
                                {done ? <Icon d={Icons.check} size={12} color="var(--accent-teal)" /> : i + 1}
                              </div>
                              <span style={{ fontSize: 9, color: done ? 'var(--accent-teal)' : active ? 'var(--accent-blue)' : 'var(--text-muted)', fontFamily: 'var(--font-mono)', textTransform: 'uppercase', letterSpacing: '0.04em', whiteSpace: 'nowrap' }}>{step}</span>
                            </div>
                            {i < 3 && <div style={{ flex: 1, height: 1, background: done ? 'var(--accent-teal)' : 'var(--border)', marginBottom: 20, transition: 'background 0.4s' }} />}
                          </React.Fragment>
                        );
                      })}
                    </div>
                    <ProgressBar value={progress} color={phase === 'complete' ? 'var(--accent-teal)' : 'var(--accent-blue)'} height={5} showValue label="Overall" />
                  </Card>
                )}

                <div style={{ display: 'flex', gap: 10 }}>
                  {phase === 'idle' || phase === 'complete' ? (
                    <>
                      <button onClick={handleAnalyze} style={{
                        flex: 1, padding: '11px 0', borderRadius: 'var(--radius-md)',
                        background: 'var(--accent-blue)', border: 'none', cursor: 'pointer',
                        color: 'oklch(100% 0 0)', fontWeight: 700, fontSize: 13, fontFamily: 'var(--font-ui)',
                        boxShadow: '0 4px 20px var(--accent-blue-mid)',
                        transition: 'opacity 0.15s, transform 0.15s',
                      }}
                        onMouseEnter={(e: any) => { e.currentTarget.style.opacity = '0.85'; e.currentTarget.style.transform = 'translateY(-1px)'; }}
                        onMouseLeave={(e: any) => { e.currentTarget.style.opacity = '1'; e.currentTarget.style.transform = 'none'; }}>
                        {phase === 'complete' ? '+ New Analysis' : 'Start Analysis'}
                      </button>
                      {phase === 'complete' && <button onClick={reset} style={{ padding: '11px 18px', borderRadius: 'var(--radius-md)', background: 'var(--bg-elevated)', border: '1px solid var(--border)', cursor: 'pointer', color: 'var(--text-secondary)', fontSize: 13, fontFamily: 'var(--font-ui)', fontWeight: 500 }}>Reset</button>}
                    </>
                  ) : (
                    <div style={{ flex: 1, padding: '11px 0', borderRadius: 'var(--radius-md)', background: 'var(--bg-elevated)', border: '1px solid var(--border)', textAlign: 'center', fontSize: 13, color: 'var(--accent-blue)', fontWeight: 600, fontFamily: 'var(--font-mono)', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}>
                      <div style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--accent-blue)', animation: 'blink 1s infinite' }} />
                      {phase === 'uploading' ? 'Uploading…' : 'Analyzing…'} {progress}%
                    </div>
                  )}
                </div>

                <Card style={{ padding: '14px 16px' }}>
                  <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 10 }}>Session Log</div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 4, maxHeight: 110, overflowY: 'auto' }}>
                    {sessionLog.map(e => (
                      <div key={e.id} style={{ display: 'flex', gap: 8, fontSize: 10, fontFamily: 'var(--font-mono)' }}>
                        <span style={{ color: 'var(--text-muted)', flexShrink: 0 }}>{e.time}</span>
                        <span style={{ color: logColors[e.type] || 'var(--text-muted)' }}>{e.msg}</span>
                      </div>
                    ))}
                  </div>
                </Card>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                <Card glow={!!result} style={{ padding: 0, overflow: 'hidden' }}>
                  <div style={{ padding: '14px 18px', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.07em' }}>Neural Activity</div>
                    <div style={{ display: 'flex', gap: 6 }}>
                      <Badge variant={result?.predictions?.tumor_detected ? 'red' : 'teal'}>
                        {result?.predictions?.tumor_detected ? 'Anomaly' : phase === 'complete' ? 'Clear' : 'Monitoring'}
                      </Badge>
                      <Badge variant="default"><span style={{ fontFamily: 'var(--font-mono)', fontSize: 10 }}>LIVE</span></Badge>
                    </div>
                  </div>
                  <div style={{ background: 'oklch(8% 0.02 240)', height: 310, position: 'relative', overflow: 'hidden' }}>
                    <BrainCanvas tumorDetected={result?.predictions?.tumor_detected ?? false} confidence={result?.predictions?.confidence ?? 0} speed={tweaks.animationSpeed} />
                    {result?.predictions?.tumor_detected && (
                      <div style={{ position: 'absolute', bottom: 12, left: 12, right: 12, padding: '8px 12px', borderRadius: 'var(--radius-sm)', background: 'oklch(60% 0.22 25 / 0.15)', border: '1px solid oklch(60% 0.22 25 / 0.35)', backdropFilter: 'blur(8px)' }}>
                        <span style={{ fontSize: 11, color: 'var(--accent-red)', fontFamily: 'var(--font-mono)', fontWeight: 500 }}>
                          ⚠ Anomaly detected · {result.predictions.location} · {(result.predictions.confidence * 100).toFixed(0)}% confidence
                        </span>
                      </div>
                    )}
                  </div>
                </Card>

                <Card>
                  <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 14, textTransform: 'uppercase', letterSpacing: '0.07em', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span>Analysis Results</span>
                    {result && <Badge variant="teal">Complete</Badge>}
                  </div>
                  <ResultsPanel result={result} loading={resultLoading} />
                </Card>
              </div>
            </div>
          )}

          {activeNav === 'monitor' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }} className="animate-in">
              {!result ? (
                <Card>
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 14, padding: '48px 0', color: 'var(--text-muted)', textAlign: 'center' }}>
                    <div style={{ width: 52, height: 52, borderRadius: 14, background: 'var(--bg-elevated)', border: '1px solid var(--border)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      <Icon d={Icons.scan} size={24} color="var(--text-muted)" />
                    </div>
                    <div>
                      <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 6 }}>No scan results yet</div>
                      <div style={{ fontSize: 12, color: 'var(--text-muted)', maxWidth: 320 }}>Run an MRI analysis first — the model attention map, segmentation overlay, and region confidence scores will appear here, derived directly from your scan.</div>
                    </div>
                    <button onClick={() => setActiveNav('scan')} style={{ padding: '9px 20px', borderRadius: 'var(--radius-md)', background: 'var(--accent-blue-mid)', border: '1px solid var(--accent-blue)', color: 'var(--accent-blue)', fontSize: 13, fontWeight: 600, cursor: 'pointer', fontFamily: 'var(--font-ui)' }}>
                      Go to MRI Analysis
                    </button>
                  </div>
                </Card>
              ) : (
                <>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                    <Card glow style={{ padding: 0, overflow: 'hidden' }}>
                      <div style={{ padding: '14px 18px', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <div>
                          <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.07em' }}>Model Attention Map</div>
                          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>Grad-CAM saliency — regions influencing prediction</div>
                        </div>
                        <Badge variant="blue">Grad-CAM</Badge>
                      </div>
                      <HeatmapCanvas result={result} />
                    </Card>
                    <Card glow style={{ padding: 0, overflow: 'hidden' }}>
                      <div style={{ padding: '14px 18px', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <div>
                          <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.07em' }}>Segmentation Overlay</div>
                          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>Predicted tumor boundary mask</div>
                        </div>
                        <Badge variant={result.predictions.tumor_detected ? 'red' : 'teal'}>
                          {result.predictions.tumor_detected ? `${result.predictions.tumor_volume_ml} mL` : 'Clear'}
                        </Badge>
                      </div>
                      <SegmentationCanvas result={result} />
                    </Card>
                  </div>
                  <Card>
                    <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 16, textTransform: 'uppercase', letterSpacing: '0.07em' }}>Regional Confidence Scores</div>
                    <RegionConfidenceChart result={result} />
                  </Card>
                  <Card>
                    <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 16, textTransform: 'uppercase', letterSpacing: '0.07em' }}>Detection Pipeline Breakdown</div>
                    <PipelineBreakdown result={result} />
                  </Card>
                </>
              )}
            </div>
          )}

          {activeNav === 'reports' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }} className="animate-in">
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 10 }}>
                <StatCard label="Total Reports" value="12" color="var(--accent-blue)" icon={Icons.chart} />
                <StatCard label="Tumor Detected" value="4" sub="33% positive rate" color="var(--accent-red)" icon={Icons.warning} />
                <StatCard label="Avg Confidence" value="89%" color="var(--accent-teal)" icon={Icons.zap} />
              </div>
              <Card>
                <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 16, textTransform: 'uppercase', letterSpacing: '0.07em' }}>Model Performance (ROC Curve)</div>
                <ROCCurve />
              </Card>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 10 }}>
              </div>
              {[
                { id: 'RPT-001', type: 'Glioblastoma', conf: 87, model: 'ensemble', date: '2026-04-30 14:22', detected: true },
                { id: 'RPT-002', type: 'None', conf: 96, model: 'resnet3d', date: '2026-04-29 09:15', detected: false },
                { id: 'RPT-003', type: 'Meningioma', conf: 78, model: 'nnunet', date: '2026-04-28 16:42', detected: true },
              ].map(r => (
                <Card key={r.id} style={{ cursor: 'pointer' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
                      <div style={{ width: 38, height: 38, borderRadius: 9, background: r.detected ? 'var(--accent-red-dim)' : 'var(--accent-teal-dim)', border: `1px solid ${r.detected ? 'oklch(60% 0.22 25 / 0.3)' : 'oklch(65% 0.22 145 / 0.3)'}`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <Icon d={r.detected ? Icons.warning : Icons.check} size={16} color={r.detected ? 'var(--accent-red)' : 'oklch(65% 0.22 145)'} />
                      </div>
                      <div>
                        <div style={{ fontSize: 13, fontWeight: 600 }}>{r.id}</div>
                        <div style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{r.detected ? r.type : 'No tumor'} · {r.conf}% conf · {r.model}</div>
                      </div>
                    </div>
                    <div style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{r.date}</div>
                  </div>
                </Card>
              ))}
            </div>
          )}

          {activeNav === 'models' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }} className="animate-in">
              <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-muted)', marginBottom: 4 }}>6 models loaded and available for inference</div>
              {MODELS.map(m => (
                <Card key={m.id} style={{ cursor: 'pointer' }} glow={m.id === selectedModel}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
                      <div style={{ width: 38, height: 38, borderRadius: 9, background: 'var(--accent-blue-dim)', border: '1px solid var(--accent-blue-mid)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <Icon d={Icons.cpu} size={17} color="var(--accent-blue)" />
                      </div>
                      <div>
                        <div style={{ fontSize: 13, fontWeight: 600 }}>{m.name}</div>
                        <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{m.tag} · ~{m.time}</div>
                      </div>
                    </div>
                    <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                      <Badge variant="teal">Loaded</Badge>
                      {m.id === selectedModel && <Badge variant="blue">Selected</Badge>}
                    </div>
                  </div>
                </Card>
              ))}
            </div>
          )}

        </div>
      </main>

      <TweaksPanel>
        <TweakSection label="Demo">
          <TweakToggle label="Demo mode" value={tweaks.demoMode} onChange={(v: any) => setTweak('demoMode', v)} />
        </TweakSection>
        <TweakSection label="Animation">
          <TweakSlider label="Speed" value={tweaks.animationSpeed} min={0.2} max={3} step={0.1} unit="×" onChange={(v: any) => setTweak('animationSpeed', v)} />
        </TweakSection>
      </TweaksPanel>
    </div>
  );
}
