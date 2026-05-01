'use client';

import React, { useState } from 'react';

// ── Slider ────────────────────────────────────────────────────────────────────
function Slider({ label, min, max, step, value, onChange, unit = "" }: any) {
  return (
    <div style={{ marginBottom: "14px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "4px" }}>
        <span style={{ fontSize: "12px", color: "#94a3b8" }}>{label}</span>
        <span style={{ fontSize: "12px", color: "#22d3ee", fontWeight: 600 }}>
          {value}{unit}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{
          width: "100%",
          accentColor: "#22d3ee",
          background: "transparent",
          cursor: "pointer",
        }}
      />
    </div>
  );
}

// ── Toggle ────────────────────────────────────────────────────────────────────
function Toggle({ label, checked, onChange }: any) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        marginBottom: "12px",
      }}
    >
      <span style={{ fontSize: "12px", color: "#94a3b8" }}>{label}</span>
      <div
        onClick={() => onChange(!checked)}
        style={{
          width: "40px",
          height: "22px",
          borderRadius: "11px",
          background: checked ? "#22d3ee" : "#334155",
          position: "relative",
          cursor: "pointer",
          transition: "background 0.25s",
          flexShrink: 0,
        }}
      >
        <div
          style={{
            position: "absolute",
            top: "3px",
            left: checked ? "21px" : "3px",
            width: "16px",
            height: "16px",
            borderRadius: "50%",
            background: "#fff",
            transition: "left 0.25s",
          }}
        />
      </div>
    </div>
  );
}

// ── Select ────────────────────────────────────────────────────────────────────
function Select({ label, options, value, onChange }: any) {
  return (
    <div style={{ marginBottom: "14px" }}>
      <div style={{ fontSize: "12px", color: "#94a3b8", marginBottom: "4px" }}>{label}</div>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        style={{
          width: "100%",
          background: "#0f172a",
          color: "#e2e8f0",
          border: "1px solid #334155",
          borderRadius: "6px",
          padding: "6px 10px",
          fontSize: "12px",
          cursor: "pointer",
          outline: "none",
        }}
      >
        {options.map((o: any) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </div>
  );
}

// ── Section ───────────────────────────────────────────────────────────────────
function Section({ title, icon, children, defaultOpen = false }: any) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div style={{ marginBottom: "16px" }}>
      <div
        onClick={() => setOpen((v: any) => !v)}
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          cursor: "pointer",
          padding: "8px 12px",
          borderRadius: "8px",
          background: "#1e293b",
          marginBottom: open ? "10px" : "0",
          userSelect: "none",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <span style={{ fontSize: "14px" }}>{icon}</span>
          <span style={{ fontSize: "12px", fontWeight: 700, color: "#cbd5e1", letterSpacing: "0.05em", textTransform: "uppercase" }}>
            {title}
          </span>
        </div>
        <span style={{ color: "#64748b", fontSize: "10px" }}>{open ? "▲" : "▼"}</span>
      </div>
      {open && <div style={{ padding: "0 4px" }}>{children}</div>}
    </div>
  );
}

// ── TweaksPanel ───────────────────────────────────────────────────────────────
export function TweaksPanel({ settings, onSettingsChange }: any) {
  const update = (key: any, value: any) => onSettingsChange({ ...settings, [key]: value });

  return (
    <div
      style={{
        height: "100%",
        overflowY: "auto",
        padding: "16px",
        scrollbarWidth: "thin",
        scrollbarColor: "#334155 transparent",
      }}
    >
      {/* ── Model ── */}
      <Section title="AI Model" icon="🤖" defaultOpen={true}>
        <Select
          label="Detection Model"
          value={settings.model}
          onChange={(v: any) => update("model", v)}
          options={[
            { value: "unet3d", label: "3D U-Net (Recommended)" },
            { value: "vit", label: "Medical ViT" },
            { value: "resnet3d", label: "ResNet-3D" },
            { value: "ensemble", label: "Ensemble (All Models)" },
          ]}
        />
        <Select
          label="Precision"
          value={settings.precision}
          onChange={(v: any) => update("precision", v)}
          options={[
            { value: "fp32", label: "FP32 (High Accuracy)" },
            { value: "fp16", label: "FP16 (Balanced)" },
            { value: "int8", label: "INT8 (Fast)" },
          ]}
        />
        <Toggle
          label="TTA (Test-Time Augmentation)"
          checked={settings.tta}
          onChange={(v: any) => update("tta", v)}
        />
      </Section>

      {/* ── Detection ── */}
      <Section title="Detection" icon="🎯">
        <Slider
          label="Confidence Threshold"
          min={0.1}
          max={1.0}
          step={0.05}
          value={settings.confidence}
          onChange={(v: any) => update("confidence", v)}
          unit=""
        />
        <Slider
          label="Min. Lesion Size (mm³)"
          min={50}
          max={5000}
          step={50}
          value={settings.minLesionSize}
          onChange={(v: any) => update("minLesionSize", v)}
          unit=" mm³"
        />
        <Toggle
          label="Multi-Class Detection"
          checked={settings.multiClass}
          onChange={(v: any) => update("multiClass", v)}
        />
        <Toggle
          label="Brain Extraction (Skull-strip)"
          checked={settings.skullStrip}
          onChange={(v: any) => update("skullStrip", v)}
        />
      </Section>

      {/* ── Preprocessing ── */}
      <Section title="Preprocessing" icon="⚙️">
        <Select
          label="Normalization"
          value={settings.normalization}
          onChange={(v: any) => update("normalization", v)}
          options={[
            { value: "zscore", label: "Z-Score" },
            { value: "minmax", label: "Min-Max" },
            { value: "histogram", label: "Histogram Equalization" },
          ]}
        />
        <Slider
          label="Gaussian Blur (sigma)"
          min={0}
          max={3}
          step={0.25}
          value={settings.gaussianBlur}
          onChange={(v: any) => update("gaussianBlur", v)}
        />
        <Toggle
          label="Bias Field Correction"
          checked={settings.biasCorrection}
          onChange={(v: any) => update("biasCorrection", v)}
        />
        <Toggle
          label="Noise Reduction"
          checked={settings.noiseReduction}
          onChange={(v: any) => update("noiseReduction", v)}
        />
      </Section>

      {/* ── Visualization ── */}
      <Section title="Visualization" icon="🎨">
        <Select
          label="Overlay Colormap"
          value={settings.colormap}
          onChange={(v: any) => update("colormap", v)}
          options={[
            { value: "hot", label: "Hot (Red-Yellow)" },
            { value: "jet", label: "Jet (Rainbow)" },
            { value: "cool", label: "Cool (Cyan-Magenta)" },
            { value: "viridis", label: "Viridis" },
            { value: "plasma", label: "Plasma" },
          ]}
        />
        <Slider
          label="Overlay Opacity"
          min={0.1}
          max={1.0}
          step={0.05}
          value={settings.overlayOpacity}
          onChange={(v: any) => update("overlayOpacity", v)}
        />
        <Toggle
          label="Show Bounding Box"
          checked={settings.showBBox}
          onChange={(v: any) => update("showBBox", v)}
        />
        <Toggle
          label="Show Confidence Map"
          checked={settings.showConfidenceMap}
          onChange={(v: any) => update("showConfidenceMap", v)}
        />
        <Toggle
          label="3D Volume Rendering"
          checked={settings.volume3D}
          onChange={(v: any) => update("volume3D", v)}
        />
      </Section>

      {/* ── Report ── */}
      <Section title="Report" icon="📄">
        <Select
          label="Output Format"
          value={settings.reportFormat}
          onChange={(v: any) => update("reportFormat", v)}
          options={[
            { value: "pdf", label: "PDF (Clinical)" },
            { value: "html", label: "HTML (Interactive)" },
            { value: "json", label: "JSON (Machine-readable)" },
            { value: "dicom_sr", label: "DICOM SR (Structured Report)" },
          ]}
        />
        <Toggle
          label="Include Raw Probabilities"
          checked={settings.includeProbs}
          onChange={(v: any) => update("includeProbs", v)}
        />
        <Toggle
          label="HIPAA-Anonymize DICOM Tags"
          checked={settings.anonymize}
          onChange={(v: any) => update("anonymize", v)}
        />
      </Section>

      {/* ── Performance ── */}
      <Section title="Performance" icon="⚡">
        <Select
          label="Device"
          value={settings.device}
          onChange={(v: any) => update("device", v)}
          options={[
            { value: "auto", label: "Auto-detect" },
            { value: "cpu", label: "CPU" },
            { value: "cuda", label: "CUDA (NVIDIA GPU)" },
            { value: "mps", label: "MPS (Apple Silicon)" },
          ]}
        />
        <Slider
          label="Batch Size"
          min={1}
          max={16}
          step={1}
          value={settings.batchSize}
          onChange={(v: any) => update("batchSize", v)}
        />
        <Toggle
          label="Cache Processed Scans"
          checked={settings.cacheScans}
          onChange={(v: any) => update("cacheScans", v)}
        />
      </Section>
    </div>
  );
}

export const defaultTweaksSettings = {
  model: "unet3d",
  precision: "fp32",
  tta: false,
  confidence: 0.5,
  minLesionSize: 200,
  multiClass: true,
  skullStrip: true,
  normalization: "zscore",
  gaussianBlur: 0,
  biasCorrection: true,
  noiseReduction: false,
  colormap: "hot",
  overlayOpacity: 0.65,
  showBBox: true,
  showConfidenceMap: false,
  volume3D: false,
  reportFormat: "pdf",
  includeProbs: false,
  anonymize: true,
  device: "auto",
  batchSize: 4,
  cacheScans: true,
};
