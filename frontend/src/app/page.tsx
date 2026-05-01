'use client';

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { TweaksPanel, defaultTweaksSettings } from '../components/TweaksPanel';
import { useEnhancedWebSocket } from '../lib/enhanced-websocket';
import { toast } from 'react-hot-toast';

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────
const TUMOR_TYPES: any = {
  glioma:      { label: "Glioma",         color: "#f87171", risk: "High" },
  meningioma:  { label: "Meningioma",     color: "#fb923c", risk: "Medium" },
  pituitary:   { label: "Pituitary",      color: "#facc15", risk: "Low-Medium" },
  no_tumor:    { label: "No Tumor",       color: "#4ade80", risk: "None" },
};

const STEPS = [
  { id: "preprocess",  label: "Preprocessing",       icon: "⚙️" },
  { id: "skull_strip", label: "Skull Stripping",      icon: "🧠" },
  { id: "normalize",   label: "Normalization",        icon: "📐" },
  { id: "inference",   label: "Model Inference",      icon: "🤖" },
  { id: "postprocess", label: "Post-processing",      icon: "🔬" },
  { id: "report",      label: "Report Generation",    icon: "📄" },
];

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
function formatBytes(bytes: number) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(2) + " MB";
}

// ─────────────────────────────────────────────────────────────────────────────
// Header
// ─────────────────────────────────────────────────────────────────────────────
function Header({ rightSlot }: any) {
  return (
    <header style={{
      display: "flex", alignItems: "center", justifyContent: "space-between",
      padding: "0 24px", height: "56px",
      background: "linear-gradient(90deg, #0f172a 0%, #0c1a35 100%)",
      borderBottom: "1px solid rgba(34,211,238,.15)",
      flexShrink: 0,
      zIndex: 10,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
        <div style={{
          width: "34px", height: "34px", borderRadius: "50%",
          background: "radial-gradient(circle, #0ea5e9 0%, #0c4a6e 100%)",
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: "16px", flexShrink: 0,
        }}>🧠</div>
        <div>
          <div style={{ fontSize: "16px", fontWeight: 800, letterSpacing: "0.03em",
            background: "linear-gradient(90deg,#38bdf8,#22d3ee,#4ade80)",
            WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
            NeuroScan
          </div>
          <div style={{ fontSize: "10px", color: "#64748b", letterSpacing: "0.08em" }}>
            BRAIN MRI TUMOR DETECTOR
          </div>
        </div>
      </div>

      <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
        {[
          { dot: "#4ade80", label: "AI Ready" },
          { dot: "#22d3ee", label: "DICOM Support" },
          { dot: "#f59e0b", label: "HIPAA Compliant" },
        ].map(({ dot, label }) => (
          <div key={label} style={{
            display: "flex", alignItems: "center", gap: "5px",
            padding: "3px 10px", borderRadius: "20px",
            background: "rgba(255,255,255,.04)", border: "1px solid rgba(255,255,255,.08)",
            fontSize: "11px", color: "#94a3b8",
          }}>
            <span style={{ width: "6px", height: "6px", borderRadius: "50%", background: dot, display: "inline-block" }} />
            {label}
          </div>
        ))}
        {rightSlot}
      </div>
    </header>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Upload Zone
// ─────────────────────────────────────────────────────────────────────────────
function UploadZone({ onFile }: any) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrop = (e: any) => {
    e.preventDefault(); setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) onFile(file);
  };

  return (
    <div
      className={dragging ? "dropzone-active" : ""}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
      style={{
        border: "2px dashed rgba(34,211,238,.3)",
        borderRadius: "12px",
        padding: "36px 20px",
        textAlign: "center",
        cursor: "pointer",
        transition: "all .2s",
        background: "rgba(15,23,42,.6)",
      }}
    >
      <input ref={inputRef} type="file"
        accept=".dcm,.nii,.nii.gz,.png,.jpg,.jpeg,.tiff,.bmp"
        style={{ display: "none" }}
        onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])}
      />
      <div style={{ fontSize: "40px", marginBottom: "12px" }}>🫁</div>
      <div style={{ fontSize: "14px", fontWeight: 600, color: "#cbd5e1", marginBottom: "6px" }}>
        Drop MRI scan here
      </div>
      <div style={{ fontSize: "11px", color: "#64748b", marginBottom: "12px" }}>
        DICOM · NIfTI · PNG · JPEG · TIFF
      </div>
      <button style={{
        padding: "7px 20px",
        borderRadius: "20px",
        background: "linear-gradient(90deg,#0ea5e9,#22d3ee)",
        border: "none", color: "#fff",
        fontSize: "12px", fontWeight: 600,
      }}>
        Browse Files
      </button>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// File Card
// ─────────────────────────────────────────────────────────────────────────────
function FileCard({ file, onRemove }: any) {
  const [url, setUrl] = useState<string | null>(null);

  useEffect(() => {
    if (file.type.startsWith("image/")) {
      const u = URL.createObjectURL(file);
      setUrl(u);
      return () => URL.revokeObjectURL(u);
    }
  }, [file]);

  return (
    <div className="fade-in" style={{
      border: "1px solid rgba(34,211,238,.2)",
      borderRadius: "10px",
      overflow: "hidden",
      background: "rgba(15,23,42,.8)",
    }}>
      {url && (
        <div style={{ position: "relative", height: "160px", background: "#000" }}>
          <img src={url} alt="MRI preview" style={{ width: "100%", height: "100%", objectFit: "contain" }} />
          <div style={{
            position: "absolute", inset: 0,
            background: "linear-gradient(to bottom, transparent 60%, rgba(2,8,23,.8))",
          }} />
        </div>
      )}
      {!url && (
        <div style={{ height: "80px", background: "rgba(15,23,42,.5)", display: "flex",
          alignItems: "center", justifyContent: "center", fontSize: "32px" }}>📂</div>
      )}
      <div style={{ padding: "10px 12px" }}>
        <div className="truncate" style={{ fontSize: "12px", fontWeight: 600, color: "#e2e8f0", marginBottom: "4px" }}>
          {file.name}
        </div>
        <div style={{ fontSize: "11px", color: "#64748b", display: "flex", justifyContent: "space-between" }}>
          <span>{formatBytes(file.size)}</span>
          <button onClick={onRemove} style={{
            background: "none", border: "none", color: "#f87171",
            fontSize: "11px", cursor: "pointer",
          }}>✕ Remove</button>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Progress Overlay
// ─────────────────────────────────────────────────────────────────────────────
function AnalysisProgress({ currentStep, percent }: any) {
  return (
    <div className="fade-in" style={{
      background: "rgba(15,23,42,.95)",
      border: "1px solid rgba(34,211,238,.2)",
      borderRadius: "12px",
      padding: "24px",
    }}>
      <div style={{ textAlign: "center", marginBottom: "20px" }}>
        <div className="pulse" style={{ fontSize: "40px", marginBottom: "8px" }}>🔬</div>
        <div style={{ fontSize: "14px", fontWeight: 700, color: "#22d3ee" }}>Analyzing MRI Scan</div>
        <div style={{ fontSize: "11px", color: "#64748b", marginTop: "2px" }}>{percent}% complete</div>
      </div>

      <div style={{ height: "8px", background: "#1e293b", borderRadius: "4px", marginBottom: "20px", overflow: "hidden" }}>
        <div
          className="progress-stripe"
          style={{
            height: "100%",
            width: `${percent}%`,
            background: "linear-gradient(90deg,#0ea5e9,#22d3ee)",
            borderRadius: "4px",
            transition: "width .5s ease",
          }}
        />
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
        {STEPS.map((s, i) => {
          const done = i < currentStep;
          const active = i === currentStep;
          return (
            <div key={s.id} style={{
              display: "flex", alignItems: "center", gap: "10px",
              opacity: done || active ? 1 : .35,
            }}>
              <div style={{
                width: "24px", height: "24px", borderRadius: "50%", flexShrink: 0,
                background: done ? "#4ade80" : active ? "#22d3ee" : "#1e293b",
                display: "flex", alignItems: "center", justifyContent: "center",
                fontSize: "11px",
              }}>
                {done ? "✓" : active ? <span className="pulse">{s.icon}</span> : s.icon}
              </div>
              <span style={{
                fontSize: "12px",
                color: done ? "#4ade80" : active ? "#22d3ee" : "#64748b",
                fontWeight: active ? 700 : 400,
              }}>{s.label}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Result Card
// ─────────────────────────────────────────────────────────────────────────────
function ResultCard({ result }: any) {
  const info = TUMOR_TYPES[result.type] || TUMOR_TYPES.no_tumor;
  const pct = Math.round(result.confidence * 100);
  const isDetected = result.type !== "no_tumor";

  return (
    <div className={`fade-in ${isDetected ? "glow-red" : "glow-green"}`} style={{
      border: `1px solid ${info.color}40`,
      borderRadius: "12px",
      background: "rgba(15,23,42,.9)",
      overflow: "hidden",
    }}>
      <div style={{
        padding: "16px 20px",
        background: `linear-gradient(90deg, ${info.color}18, transparent)`,
        borderBottom: `1px solid ${info.color}30`,
        display: "flex", alignItems: "center", gap: "12px",
      }}>
        <div style={{
          width: "44px", height: "44px", borderRadius: "50%",
          background: `${info.color}20`,
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: "22px", flexShrink: 0,
        }}>
          {isDetected ? "⚠️" : "✅"}
        </div>
        <div>
          <div style={{ fontSize: "16px", fontWeight: 800, color: info.color }}>{info.label}</div>
          <div style={{ fontSize: "11px", color: "#94a3b8" }}>
            Risk level: <span style={{ color: info.color, fontWeight: 600 }}>{info.risk}</span>
          </div>
        </div>
        <div style={{ marginLeft: "auto", textAlign: "right" }}>
          <div style={{ fontSize: "28px", fontWeight: 900, color: info.color }}>{pct}%</div>
          <div style={{ fontSize: "10px", color: "#64748b" }}>confidence</div>
        </div>
      </div>

      <div style={{ padding: "16px 20px", display: "flex", flexDirection: "column", gap: "10px" }}>
        {isDetected && (
          <>
            <Detail icon="📍" label="Location" value={result.location} />
            <Detail icon="📏" label="Estimated Volume" value={`${result.volume.toLocaleString()} mm³`} />
          </>
        )}
        <Detail icon="🎲" label="Uncertainty" value={`±${result.uncertainty}`} />
        <Detail icon="⏱️" label="Processing Time" value={`${result.processing_time}s`} />

        <div style={{ marginTop: "4px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "4px" }}>
            <span style={{ fontSize: "11px", color: "#64748b" }}>Confidence</span>
            <span style={{ fontSize: "11px", color: info.color }}>{pct}%</span>
          </div>
          <div style={{ height: "6px", background: "#1e293b", borderRadius: "3px", overflow: "hidden" }}>
            <div style={{ height: "100%", width: `${pct}%`, background: info.color, borderRadius: "3px", transition: "width 1s ease" }} />
          </div>
        </div>

        <div style={{
          marginTop: "8px", padding: "10px 12px",
          background: "rgba(248,113,113,.06)", border: "1px solid rgba(248,113,113,.15)",
          borderRadius: "8px", fontSize: "10px", color: "#fca5a5", lineHeight: "1.5",
        }}>
          ⚕️ <strong>Clinical Advisory:</strong> This AI-generated result is for research assistance
          only. A qualified neuroradiologist must review all findings before clinical decisions are made.
        </div>

        <div style={{ display: "flex", gap: "8px", marginTop: "4px" }}>
          {["📄 Export Report", "🖼 Save Overlay", "📋 Copy JSON"].map((label) => (
            <button key={label} style={{
              flex: 1, padding: "7px 4px",
              background: "rgba(255,255,255,.04)",
              border: "1px solid rgba(255,255,255,.08)",
              borderRadius: "8px",
              color: "#94a3b8", fontSize: "11px",
              transition: "all .2s",
            }}
            >
              {label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function Detail({ icon, label, value }: any) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
      <span style={{ fontSize: "12px", color: "#64748b" }}>{icon} {label}</span>
      <span style={{ fontSize: "12px", color: "#cbd5e1", fontWeight: 600 }}>{value}</span>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Left sidebar – controls
// ─────────────────────────────────────────────────────────────────────────────
function ControlPanel({ file, onFile, onRemove, onAnalyze, analyzing, result, settings }: any) {
  const canAnalyze = !!file && !analyzing;

  return (
    <div style={{
      width: "300px", flexShrink: 0,
      borderRight: "1px solid rgba(255,255,255,.06)",
      display: "flex", flexDirection: "column",
      background: "rgba(10,15,30,.6)",
    }}>
      <div style={{ padding: "16px 20px", borderBottom: "1px solid rgba(255,255,255,.06)" }}>
        <div style={{ fontSize: "11px", color: "#64748b", fontWeight: 700, letterSpacing: ".08em", textTransform: "uppercase" }}>
          Scan Upload
        </div>
      </div>

      <div style={{ padding: "16px 20px", overflowY: "auto", flex: 1, display: "flex", flexDirection: "column", gap: "16px" }}>
        {!file && <UploadZone onFile={onFile} />}
        {file && <FileCard file={file} onRemove={onRemove} />}

        <div style={{
          padding: "10px 14px",
          background: "rgba(14,165,233,.06)",
          border: "1px solid rgba(14,165,233,.15)",
          borderRadius: "8px",
          fontSize: "11px", color: "#94a3b8",
          lineHeight: "1.5",
        }}>
          🤖 Model: <span style={{ color: "#38bdf8", fontWeight: 600 }}>
            {settings.model === "unet3d" ? "3D U-Net" :
             settings.model === "vit" ? "Medical ViT" :
             settings.model === "resnet3d" ? "ResNet-3D" : "Ensemble"}
          </span>
          <br />
          🎯 Threshold: <span style={{ color: "#22d3ee", fontWeight: 600 }}>{(settings.confidence * 100).toFixed(0)}%</span>
          {settings.tta && <><br />✨ TTA: <span style={{ color: "#a78bfa", fontWeight: 600 }}>Enabled</span></>}
        </div>

        <button
          onClick={onAnalyze}
          disabled={!canAnalyze}
          style={{
            width: "100%", padding: "13px",
            background: canAnalyze
              ? "linear-gradient(90deg,#0ea5e9,#22d3ee)"
              : "rgba(255,255,255,.08)",
            border: "none", borderRadius: "10px",
            color: canAnalyze ? "#fff" : "#475569",
            fontSize: "14px", fontWeight: 700,
            cursor: canAnalyze ? "pointer" : "not-allowed",
            transition: "all .2s",
            boxShadow: canAnalyze ? "0 0 20px rgba(34,211,238,.3)" : "none",
          }}
        >
          {analyzing ? "⏳ Analyzing…" : "🔬 Run Analysis"}
        </button>

        {result && !analyzing && <ResultCard result={result} />}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Centre – viewer
// ─────────────────────────────────────────────────────────────────────────────
function ViewerPanel({ file, analyzing, currentStep, percent, result }: any) {
  const [url, setUrl] = useState<string | null>(null);

  useEffect(() => {
    if (file && file.type.startsWith("image/")) {
      const u = URL.createObjectURL(file);
      setUrl(u);
      return () => URL.revokeObjectURL(u);
    } else {
      setUrl(null);
    }
  }, [file]);

  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", position: "relative", overflow: "hidden" }}>
      <div style={{
        display: "flex", alignItems: "center", gap: "8px",
        padding: "10px 16px",
        borderBottom: "1px solid rgba(255,255,255,.06)",
        background: "rgba(10,15,30,.4)",
        flexShrink: 0,
      }}>
        {["⚡ Axial", "⚡ Coronal", "⚡ Sagittal", "🔲 3D Volume"].map((v) => (
          <button key={v} style={{
            padding: "5px 12px", borderRadius: "6px",
            background: v.includes("Axial") ? "rgba(34,211,238,.12)" : "rgba(255,255,255,.04)",
            border: v.includes("Axial") ? "1px solid rgba(34,211,238,.3)" : "1px solid rgba(255,255,255,.06)",
            color: v.includes("Axial") ? "#22d3ee" : "#64748b",
            fontSize: "11px", fontWeight: 600,
          }}>{v}</button>
        ))}
        <div style={{ marginLeft: "auto", display: "flex", gap: "6px" }}>
          {["🔍+", "🔍−", "⟲ Reset"].map((v) => (
            <button key={v} style={{
              padding: "5px 10px", borderRadius: "6px",
              background: "rgba(255,255,255,.04)",
              border: "1px solid rgba(255,255,255,.06)",
              color: "#64748b", fontSize: "11px",
            }}>{v}</button>
          ))}
        </div>
      </div>

      <div style={{ flex: 1, position: "relative", display: "flex", alignItems: "center", justifyContent: "center", background: "#020817" }}>
        {!file && !analyzing && (
          <div style={{ textAlign: "center", color: "#334155" }}>
            <div style={{ fontSize: "64px", marginBottom: "16px", opacity: .4 }}>🧠</div>
            <div style={{ fontSize: "14px", fontWeight: 600 }}>No scan loaded</div>
            <div style={{ fontSize: "12px", marginTop: "6px" }}>Upload an MRI file to begin</div>
          </div>
        )}

        {url && !analyzing && (
          <div style={{ position: "relative", maxWidth: "90%", maxHeight: "90%" }}>
            <img src={url} alt="MRI scan"
              style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain", display: "block" }} />
            <div style={{
              position: "absolute", inset: 0, pointerEvents: "none",
              backgroundImage: "linear-gradient(rgba(34,211,238,.03) 1px, transparent 1px), linear-gradient(90deg, rgba(34,211,238,.03) 1px, transparent 1px)",
              backgroundSize: "30px 30px",
            }} />
            {result && result.type !== "no_tumor" && (
              <div style={{
                position: "absolute", top: "20%", left: "30%",
                width: "40%", height: "35%",
                border: `2px solid ${TUMOR_TYPES[result.type].color}`,
                borderRadius: "4px",
                boxShadow: `0 0 12px ${TUMOR_TYPES[result.type].color}60`,
                pointerEvents: "none",
              }}>
                <div style={{
                  position: "absolute", top: "-20px", left: 0,
                  background: TUMOR_TYPES[result.type].color,
                  color: "#000", fontSize: "10px", fontWeight: 700,
                  padding: "2px 6px", borderRadius: "3px",
                }}>
                  {TUMOR_TYPES[result.type].label} {Math.round(result.confidence * 100)}%
                </div>
              </div>
            )}
          </div>
        )}

        {analyzing && (
          <div style={{ width: "100%", maxWidth: "420px", padding: "24px" }}>
            <AnalysisProgress currentStep={currentStep} percent={percent} />
          </div>
        )}

        {file && (
          <>
            {[
              { pos: { top: 12, left: 12 },   items: ["R", "NeuroScan v2.0"] },
              { pos: { top: 12, right: 12 },  items: ["A", "Axial +0.0mm"] },
              { pos: { bottom: 12, left: 12 }, items: ["L", "WW: 80 / WL: 40"] },
              { pos: { bottom: 12, right: 12 }, items: ["P", "FOV: 240mm"] },
            ].map(({ pos, items }, idx) => (
              <div key={idx} style={{
                position: "absolute", ...pos,
                display: "flex", flexDirection: "column",
                gap: "2px",
              }}>
                {items.map((t) => (
                  <span key={t} style={{ fontSize: "10px", color: "#22d3ee", opacity: .7, fontFamily: "monospace" }}>{t}</span>
                ))}
              </div>
            ))}
          </>
        )}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Right sidebar – tweaks
// ─────────────────────────────────────────────────────────────────────────────
function TweaksBar({ settings, onSettingsChange }: any) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div style={{
      width: collapsed ? "44px" : "280px",
      flexShrink: 0,
      borderLeft: "1px solid rgba(255,255,255,.06)",
      background: "rgba(10,15,30,.6)",
      display: "flex",
      flexDirection: "column",
      transition: "width .25s ease",
      overflow: "hidden",
    }}>
      <div style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: collapsed ? "16px 10px" : "16px 20px",
        borderBottom: "1px solid rgba(255,255,255,.06)",
        flexShrink: 0,
      }}>
        {!collapsed && (
          <div style={{ fontSize: "11px", color: "#64748b", fontWeight: 700, letterSpacing: ".08em", textTransform: "uppercase" }}>
            ⚙️ Tweaks
          </div>
        )}
        <button
          onClick={() => setCollapsed(v => !v)}
          style={{
            background: "none", border: "none", color: "#64748b",
            fontSize: "16px", cursor: "pointer", padding: 0,
            marginLeft: collapsed ? 0 : "auto",
          }}
          title={collapsed ? "Expand tweaks" : "Collapse tweaks"}
        >
          {collapsed ? "◀" : "▶"}
        </button>
      </div>
      {!collapsed && (
        <div style={{ flex: 1, overflow: "hidden" }}>
          <TweaksPanel settings={settings} onSettingsChange={onSettingsChange} />
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Root App
// ─────────────────────────────────────────────────────────────────────────────
export default function HomePage() {
  const websocket = useEnhancedWebSocket();
  const [file, setFile] = useState<File | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [currentStep, setCurrentStep] = useState(-1);
  const [percent, setPercent] = useState(0);
  const [result, setResult] = useState<any>(null);
  const [settings, setSettings] = useState(defaultTweaksSettings);
  const [isConnected, setIsConnected] = useState(false);
  const [analysisId, setAnalysisId] = useState<string | null>(null);

  useEffect(() => {
    const userId = `user_${Math.random().toString(36).substr(2, 9)}`;
    websocket.connect(userId).then(connected => {
      setIsConnected(connected);
    });

    const unsubscribe = websocket.onAnalysisUpdate((data: any) => {
      if (data.type === 'analysis_progress' || (data.type === 'analysis_update' && data.status === 'processing')) {
        const progress = data.progress || 0;
        setPercent(progress);
        const stepIndex = Math.floor((progress / 100) * STEPS.length);
        setCurrentStep(Math.min(stepIndex, STEPS.length - 1));
      } else if (data.type === 'analysis_complete' || (data.type === 'analysis_update' && data.status === 'completed')) {
        setResult(data.results);
        setAnalyzing(false);
        setCurrentStep(-1);
        toast.success('Analysis complete!');
      } else if (data.type === 'analysis_error' || (data.type === 'analysis_update' && data.status === 'failed')) {
        setAnalyzing(false);
        setCurrentStep(-1);
        toast.error(`Analysis failed: ${data.message || data.error}`);
      }
    });

    return () => {
      unsubscribe();
      websocket.disconnect();
    };
  }, [websocket]);

  const handleFile = useCallback((f: File) => {
    setFile(f);
    setResult(null);
    setAnalyzing(false);
  }, []);

  const handleRemove = useCallback(() => {
    setFile(null);
    setResult(null);
    setAnalyzing(false);
  }, []);

  const handleAnalyze = useCallback(async () => {
    if (!file || analyzing) return;

    setAnalyzing(true);
    setResult(null);
    setCurrentStep(0);
    setPercent(0);

    try {
      const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';
      const formData = new FormData();
      formData.append('file', file);
      formData.append('model', settings.model);

      const response = await fetch(`${API_BASE_URL}/api/v1/analysis/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Upload failed');
      const data = await response.json();
      setAnalysisId(data.analysis_id);
    } catch (err: any) {
      toast.error(`Error: ${err.message}`);
      setAnalyzing(false);
    }
  }, [file, analyzing, settings]);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", overflow: "hidden" }}>
      <Header rightSlot={
        <div style={{
          display: "flex", alignItems: "center", gap: "5px",
          padding: "3px 10px", borderRadius: "20px",
          background: "rgba(255,255,255,.04)", border: "1px solid rgba(255,255,255,.08)",
          fontSize: "11px", color: isConnected ? "#4ade80" : "#f87171",
        }}>
          <span style={{ width: "6px", height: "6px", borderRadius: "50%", background: isConnected ? "#4ade80" : "#f87171", display: "inline-block" }} />
          {isConnected ? "WS Active" : "WS Offline"}
        </div>
      } />

      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
        <ControlPanel
          file={file}
          onFile={handleFile}
          onRemove={handleRemove}
          onAnalyze={handleAnalyze}
          analyzing={analyzing}
          result={result}
          settings={settings}
        />

        <ViewerPanel
          file={file}
          analyzing={analyzing}
          currentStep={currentStep}
          percent={percent}
          result={result}
        />

        <TweaksBar settings={settings} onSettingsChange={setSettings} />
      </div>
    </div>
  );
}
