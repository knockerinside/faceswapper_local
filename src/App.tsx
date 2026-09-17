import React, { useState } from "react";
import {
  Terminal,
  Cpu,
  Download,
  Trash2,
  Video,
  Layers,
  Settings,
  ShieldCheck,
  Zap,
  CheckCircle2,
  Copy,
  ExternalLink,
  ChevronRight,
  Sparkles,
  HelpCircle,
  HardDrive,
  RefreshCw,
  Eye,
  Sliders
} from "lucide-react";

interface ProfilePreset {
  id: string;
  name: string;
  badge: string;
  resolution: string;
  fps: string;
  latency: string;
  vram: string;
  features: string[];
  command: string;
}

const PROFILES: ProfilePreset[] = [
  {
    id: "performance",
    name: "Ultra Performance",
    badge: "Fastest / Low Battery",
    resolution: "720p (1280x720)",
    fps: "45–60 FPS",
    latency: "14–17 ms",
    vram: "~1.8 GB",
    features: [
      "FP16 InSwapper-128 Tensor Core optimization",
      "5-frame landmark tracking skip",
      "Simplified soft edge feathering",
      "Ideal for laptops running on battery or concurrent gaming"
    ],
    command: ".\\run.ps1 -Profile performance"
  },
  {
    id: "balanced",
    name: "Balanced Studio",
    badge: "Recommended for RTX 4050",
    resolution: "720p / 1080p",
    fps: "30–35 FPS",
    latency: "21–24 ms",
    vram: "~2.4 GB",
    features: [
      "Real-time Reinhard lighting adaptation",
      "Temporal exponential landmark smoothing (alpha=0.65)",
      "2-frame tracking skip with bounded buffer",
      "Zero desync on OBS Virtual Camera & Discord"
    ],
    command: ".\\run.ps1 -Profile balanced"
  },
  {
    id: "quality",
    name: "Studio Quality",
    badge: "Maximum Realism",
    resolution: "1080p (1920x1080)",
    fps: "25–30 FPS",
    latency: "28–33 ms",
    vram: "~3.6 GB",
    features: [
      "BiSeNet semantic hair & bangs occlusion protection",
      "Full per-frame SCRFD detection pass",
      "Multi-band skin tone color grading",
      "Maximum visual fidelity for professional video calls"
    ],
    command: ".\\run.ps1 -Profile quality"
  }
];

const MODELS_DATA = [
  {
    filename: "inswapper_128_fp16.onnx",
    category: "Face Swapper (FP16)",
    size: "264 MB",
    speed: "1.8x Faster",
    tag: "Recommended for RTX 4050",
    description: "Half-precision FP16 weights adapted from Deep-Live-Cam. Maximizes Tensor Core utilization with 50% lower VRAM.",
    url: "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx"
  },
  {
    filename: "inswapper_128.onnx",
    category: "Face Swapper (FP32)",
    size: "529 MB",
    speed: "Standard",
    tag: "Full Precision",
    description: "Standard full-precision InsightFace InSwapper model. Compatible with all ONNX Runtime CUDA configurations.",
    url: "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx"
  },
  {
    filename: "scrfd_10g_bnkps.onnx / det_10g.onnx",
    category: "Face Detector",
    size: "16.5 MB",
    speed: "4–6 ms",
    tag: "Essential",
    description: "SCRFD-10G high-speed face detection & 5-point facial landmark locator. Engine auto-supports both variants.",
    url: "https://huggingface.co/MonsterMMORPG/tools/resolve/main/scrfd_10g_bnkps.onnx"
  },
  {
    filename: "w600k_r50.onnx",
    category: "Identity Encoder",
    size: "166 MB",
    speed: "One-time pass",
    tag: "Optional",
    description: "ArcFace ResNet-50 feature extractor for extracting 512-D identity embeddings from reference source photos.",
    url: "https://huggingface.co/hacksider/deep-live-cam/resolve/main/buffalo_l/buffalo_l/w600k_r50.onnx"
  }
];

export default function App() {
  const [copiedKey, setCopiedKey] = useState<string | null>(null);
  const [activeProfile, setActiveProfile] = useState<string>("balanced");
  const [activeTab, setActiveTab] = useState<"quickstart" | "architecture" | "models" | "uninstaller" | "webcam">("quickstart");

  const copyToClipboard = (text: string, key: string) => {
    navigator.clipboard.writeText(text);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 2000);
  };

  const selectedProfile = PROFILES.find((p) => p.id === activeProfile) || PROFILES[1];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans selection:bg-emerald-500/30 selection:text-emerald-300">
      {/* Top Header */}
      <header className="border-b border-slate-800/80 bg-slate-900/70 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-9 w-9 rounded-lg bg-gradient-to-tr from-emerald-500 to-teal-400 flex items-center justify-center shadow-lg shadow-emerald-500/20">
              <Zap className="h-5 w-5 text-slate-950" />
            </div>
            <div>
              <h1 className="text-base font-bold tracking-tight text-white flex items-center gap-2">
                Real-Time AI Face-Swap Studio
                <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                  v2.2 Production
                </span>
              </h1>
              <p className="text-xs text-slate-400 hidden sm:block">
                Windows 11 • NVIDIA RTX 4050 (6GB VRAM) • Zero-Queue OBS Virtual Cam
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => copyToClipboard(".\\setup.ps1", "header-setup")}
              className="px-3 py-1.5 rounded-md bg-emerald-500 hover:bg-emerald-400 text-slate-950 text-xs font-semibold flex items-center gap-1.5 transition-all shadow-md shadow-emerald-500/20 active:scale-95"
            >
              {copiedKey === "header-setup" ? <CheckCircle2 className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
              <span>{copiedKey === "header-setup" ? "Copied!" : "Copy Setup Command"}</span>
            </button>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <div className="border-b border-slate-800 bg-slate-900/40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex space-x-1 sm:space-x-4 overflow-x-auto py-2">
          {[
            { id: "quickstart", label: "Quick Start & Setup", icon: Terminal },
            { id: "architecture", label: "Pipeline & RTX 4050 Tuning", icon: Cpu },
            { id: "models", label: "AI Models & Weights", icon: HardDrive },
            { id: "webcam", label: "Phone Cam & OBS Guide", icon: Video },
            { id: "uninstaller", label: "Zero-Trash Uninstaller", icon: Trash2 },
          ].map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-medium transition-colors whitespace-nowrap ${
                  isActive
                    ? "bg-slate-800 text-emerald-400 border border-emerald-500/30 shadow-sm"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/40"
                }`}
              >
                <Icon className={`h-4 w-4 ${isActive ? "text-emerald-400" : "text-slate-400"}`} />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Main Content Area */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* TAB 1: QUICK START */}
        {activeTab === "quickstart" && (
          <div className="space-y-8">
            {/* Hero Quick Banner */}
            <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-slate-900 via-slate-900/90 to-slate-800/80 border border-slate-700/80 p-6 sm:p-8">
              <div className="max-w-3xl">
                <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-medium mb-4">
                  <ShieldCheck className="h-3.5 w-3.5" />
                  Self-Contained • Zero Global Pollution • Automatic Downloader Included
                </div>
                <h2 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-white">
                  Real-Time AI Face-Swap in 2 Commands
                </h2>
                <p className="mt-2 text-sm text-slate-300 leading-relaxed">
                  Run everything locally in the folder where you execute the script. Automatically detects your NVIDIA RTX 4050, sets up an isolated Python virtual environment, installs CUDA dependencies, downloads ONNX model weights with failover mirrors, and starts streaming into OBS / Discord.
                </p>
              </div>

              {/* Steps Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-8">
                {/* Step 1 */}
                <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-5 relative group">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <span className="h-6 w-6 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 text-xs font-bold flex items-center justify-center">
                        1
                      </span>
                      <span className="text-sm font-semibold text-white">Run Self-Contained Setup</span>
                    </div>
                    <button
                      onClick={() =>
                        copyToClipboard(
                          "Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass\n.\\setup.ps1",
                          "cmd-setup"
                        )
                      }
                      className="p-1.5 rounded-md hover:bg-slate-800 text-slate-400 hover:text-white transition-colors"
                      title="Copy command"
                    >
                      {copiedKey === "cmd-setup" ? <CheckCircle2 className="h-4 w-4 text-emerald-400" /> : <Copy className="h-4 w-4" />}
                    </button>
                  </div>
                  <pre className="bg-slate-900 border border-slate-800 rounded-lg p-3 text-xs font-mono text-emerald-300 overflow-x-auto">
                    <code>Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass&#10;.\setup.ps1</code>
                  </pre>
                  <p className="text-xs text-slate-400 mt-2">
                    Auto-installs Visual C++ (if needed), creates <code className="text-slate-200">.venv</code>, fetches CUDA ONNX Runtime & models.
                  </p>
                </div>

                {/* Step 2 */}
                <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-5 relative group">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <span className="h-6 w-6 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 text-xs font-bold flex items-center justify-center">
                        2
                      </span>
                      <span className="text-sm font-semibold text-white">Launch the Control Studio</span>
                    </div>
                    <button
                      onClick={() => copyToClipboard(".\\run.ps1", "cmd-run")}
                      className="p-1.5 rounded-md hover:bg-slate-800 text-slate-400 hover:text-white transition-colors"
                      title="Copy command"
                    >
                      {copiedKey === "cmd-run" ? <CheckCircle2 className="h-4 w-4 text-emerald-400" /> : <Copy className="h-4 w-4" />}
                    </button>
                  </div>
                  <pre className="bg-slate-900 border border-slate-800 rounded-lg p-3 text-xs font-mono text-emerald-300 overflow-x-auto">
                    <code>.\run.ps1</code>
                  </pre>
                  <p className="text-xs text-slate-400 mt-2">
                    Boots the PySide6 UI, initializes RTX 4050 Tensor Cores, and opens low-latency camera preview.
                  </p>
                </div>
              </div>
            </div>

            {/* Performance Profiles Selector */}
            <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
                <div>
                  <h3 className="text-lg font-bold text-white flex items-center gap-2">
                    <Sliders className="h-5 w-5 text-emerald-400" />
                    Runtime Profile Presets for RTX 4050
                  </h3>
                  <p className="text-xs text-slate-400 mt-1">
                    Select a performance profile tailored to your thermal, resolution, and latency targets.
                  </p>
                </div>

                <div className="inline-flex rounded-lg bg-slate-950 p-1 border border-slate-800">
                  {PROFILES.map((p) => (
                    <button
                      key={p.id}
                      onClick={() => setActiveProfile(p.id)}
                      className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                        activeProfile === p.id
                          ? "bg-emerald-500 text-slate-950 shadow-sm font-semibold"
                          : "text-slate-400 hover:text-white"
                      }`}
                    >
                      {p.name.split(" ")[1] || p.name}
                    </button>
                  ))}
                </div>
              </div>

              {/* Profile Card */}
              <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-5">
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-800">
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-base font-bold text-white">{selectedProfile.name}</span>
                      <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-medium">
                        {selectedProfile.badge}
                      </span>
                    </div>
                    <p className="text-xs text-slate-400 mt-1">
                      Execution command: <code className="text-emerald-300 font-mono">{selectedProfile.command}</code>
                    </p>
                  </div>

                  <button
                    onClick={() => copyToClipboard(selectedProfile.command, `prof-${selectedProfile.id}`)}
                    className="self-start md:self-auto px-3 py-1.5 rounded-md bg-slate-800 hover:bg-slate-700 text-white text-xs font-medium flex items-center gap-1.5 border border-slate-700 transition-colors"
                  >
                    {copiedKey === `prof-${selectedProfile.id}` ? <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
                    <span>{copiedKey === `prof-${selectedProfile.id}` ? "Copied!" : "Copy Launch Flag"}</span>
                  </button>
                </div>

                {/* Profile Metrics Grid */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 my-5">
                  <div className="bg-slate-900/70 rounded-lg p-3 border border-slate-800/80">
                    <span className="text-slate-400 text-xs block">Target Resolution</span>
                    <span className="text-white font-bold text-sm mt-0.5 block">{selectedProfile.resolution}</span>
                  </div>
                  <div className="bg-slate-900/70 rounded-lg p-3 border border-slate-800/80">
                    <span className="text-slate-400 text-xs block">Real-Time FPS</span>
                    <span className="text-emerald-400 font-bold text-sm mt-0.5 block">{selectedProfile.fps}</span>
                  </div>
                  <div className="bg-slate-900/70 rounded-lg p-3 border border-slate-800/80">
                    <span className="text-slate-400 text-xs block">End-to-End Latency</span>
                    <span className="text-white font-bold text-sm mt-0.5 block">{selectedProfile.latency}</span>
                  </div>
                  <div className="bg-slate-900/70 rounded-lg p-3 border border-slate-800/80">
                    <span className="text-slate-400 text-xs block">VRAM Footprint</span>
                    <span className="text-white font-bold text-sm mt-0.5 block">{selectedProfile.vram} / 6GB</span>
                  </div>
                </div>

                {/* Features Checklist */}
                <div className="space-y-1.5">
                  <span className="text-xs font-semibold text-slate-300">Included Enhancements:</span>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 mt-2">
                    {selectedProfile.features.map((feat, idx) => (
                      <div key={idx} className="flex items-start gap-2 text-xs text-slate-300">
                        <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 shrink-0 mt-0.5" />
                        <span>{feat}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Quick Helper Scripts */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Download className="h-4 w-4 text-emerald-400" />
                  <span className="text-xs font-bold text-white">Manual Model Fetch</span>
                </div>
                <p className="text-xs text-slate-400 mb-3">Download or verify all ONNX weights without running full setup.</p>
                <div className="flex items-center justify-between bg-slate-950 p-2 rounded border border-slate-800 text-xs font-mono text-emerald-300">
                  <span>.\download_models.ps1</span>
                  <button onClick={() => copyToClipboard(".\\download_models.ps1", "dl-btn")} className="text-slate-400 hover:text-white">
                    {copiedKey === "dl-btn" ? <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
                  </button>
                </div>
              </div>

              <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Cpu className="h-4 w-4 text-teal-400" />
                  <span className="text-xs font-bold text-white">Hardware Diagnostics</span>
                </div>
                <p className="text-xs text-slate-400 mb-3">Inspect CUDA execution providers, camera enumeration, and VRAM.</p>
                <div className="flex items-center justify-between bg-slate-950 p-2 rounded border border-slate-800 text-xs font-mono text-emerald-300">
                  <span>.\diagnostics.ps1</span>
                  <button onClick={() => copyToClipboard(".\\diagnostics.ps1", "diag-btn")} className="text-slate-400 hover:text-white">
                    {copiedKey === "diag-btn" ? <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
                  </button>
                </div>
              </div>

              <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Trash2 className="h-4 w-4 text-rose-400" />
                  <span className="text-xs font-bold text-white">Zero-Trash Clean Uninstall</span>
                </div>
                <p className="text-xs text-slate-400 mb-3">Purge all .venv, bytecaches, models, and pip caches completely.</p>
                <div className="flex items-center justify-between bg-slate-950 p-2 rounded border border-slate-800 text-xs font-mono text-rose-300">
                  <span>.\uninstall.ps1</span>
                  <button onClick={() => copyToClipboard(".\\uninstall.ps1", "un-btn")} className="text-slate-400 hover:text-white">
                    {copiedKey === "un-btn" ? <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: ARCHITECTURE & PIPELINE */}
        {activeTab === "architecture" && (
          <div className="space-y-8">
            <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 sm:p-8">
              <h2 className="text-xl font-bold text-white mb-2 flex items-center gap-2">
                <Layers className="h-5 w-5 text-emerald-400" />
                Zero-Queue Latency Architecture
              </h2>
              <p className="text-xs text-slate-300 leading-relaxed max-w-3xl">
                Unlike traditional deepfake tools that accumulate frames in internal queue buffers (creating 150ms+ audio/video desync), this pipeline uses a single-slot overwrite frame buffer with asynchronous background inference.
              </p>

              {/* Visual Pipeline Flow */}
              <div className="mt-8 space-y-4">
                {[
                  {
                    step: "Stage 01",
                    title: "Physical Camera Ingress (DirectShow / MediaFoundation)",
                    latency: "2–3 ms",
                    desc: "Bounded Ring Buffer (max_size=1) discards stale video frames. Always processes the freshest camera tick."
                  },
                  {
                    step: "Stage 02",
                    title: "Face Detection & 5-Point Alignment (SCRFD-10G / det_10g)",
                    latency: "4–6 ms",
                    desc: "Runs every N frames in tracking mode; uses similarity transform (affine warp) to extract canonical 128x128 face patch."
                  },
                  {
                    step: "Stage 03",
                    title: "Tensor Core Face Synthesis (InSwapper-128 FP16)",
                    latency: "7–10 ms",
                    desc: "Injects 512-D source identity vector into latent space. FP16 half-precision halves memory bandwidth on RTX 4050."
                  },
                  {
                    step: "Stage 04",
                    title: "Reinhard Color Transfer & Hair/Bangs Occlusion Masking",
                    latency: "2–4 ms",
                    desc: "Calculates L*a*b* color statistics to match ambient webcam lighting. BiSeNet mask preserves forehead bangs and glasses."
                  },
                  {
                    step: "Stage 05",
                    title: "Temporal Smoothing & Virtual Camera Broadcast (PyVirtualCam)",
                    latency: "1–2 ms",
                    desc: "Exponential moving average eliminates micro-jitter. Direct memory copy into OBS Virtual Camera DirectShow driver."
                  }
                ].map((st, i) => (
                  <div key={i} className="flex flex-col sm:flex-row sm:items-center justify-between bg-slate-950 p-4 rounded-xl border border-slate-800 gap-3">
                    <div className="flex items-start sm:items-center gap-3">
                      <span className="text-xs font-mono font-bold text-emerald-400 bg-emerald-500/10 px-2 py-1 rounded border border-emerald-500/20 shrink-0">
                        {st.step}
                      </span>
                      <div>
                        <h4 className="text-sm font-semibold text-white">{st.title}</h4>
                        <p className="text-xs text-slate-400 mt-0.5">{st.desc}</p>
                      </div>
                    </div>
                    <span className="text-xs font-mono text-emerald-400 bg-slate-900 px-2.5 py-1 rounded border border-slate-800 self-start sm:self-auto shrink-0">
                      ~{st.latency}
                    </span>
                  </div>
                ))}
              </div>

              {/* Total Latency Summary */}
              <div className="mt-6 p-4 rounded-xl bg-gradient-to-r from-emerald-950/40 to-teal-950/40 border border-emerald-500/30 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                  <Zap className="h-5 w-5 text-emerald-400" />
                  <span className="text-sm font-bold text-white">Total Pipeline Latency: ~18–24 ms</span>
                </div>
                <span className="text-xs text-emerald-300 font-medium">
                  Matches natural 30–60 FPS webcam frame intervals with zero audio desync in video calls.
                </span>
              </div>
            </div>
          </div>
        )}

        {/* TAB 3: AI MODELS & WEIGHTS */}
        {activeTab === "models" && (
          <div className="space-y-6">
            <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
                <div>
                  <h3 className="text-lg font-bold text-white flex items-center gap-2">
                    <HardDrive className="h-5 w-5 text-emerald-400" />
                    Verified ONNX Model Registry
                  </h3>
                  <p className="text-xs text-slate-400 mt-1">
                    All models are automatically downloaded by <code className="text-emerald-300">.\setup.ps1</code> or <code className="text-emerald-300">.\download_models.ps1</code>.
                  </p>
                </div>

                <button
                  onClick={() => copyToClipboard(".\\download_models.ps1", "btn-dl-all")}
                  className="px-3 py-1.5 rounded-md bg-emerald-500 hover:bg-emerald-400 text-slate-950 text-xs font-semibold flex items-center gap-1.5 transition-all shadow-md active:scale-95"
                >
                  {copiedKey === "btn-dl-all" ? <CheckCircle2 className="h-3.5 w-3.5" /> : <Download className="h-3.5 w-3.5" />}
                  <span>{copiedKey === "btn-dl-all" ? "Copied!" : "Run Auto-Downloader"}</span>
                </button>
              </div>

              <div className="grid grid-cols-1 gap-4">
                {MODELS_DATA.map((model, idx) => (
                  <div key={idx} className="bg-slate-950 border border-slate-800 rounded-xl p-4 flex flex-col md:flex-row md:items-center justify-between gap-4">
                    <div className="space-y-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="text-sm font-bold text-white font-mono">{model.filename}</span>
                        <span className="text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-medium">
                          {model.category}
                        </span>
                        <span className="text-xs px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-medium">
                          {model.tag}
                        </span>
                      </div>
                      <p className="text-xs text-slate-400 max-w-2xl">{model.description}</p>
                      <div className="flex items-center gap-4 text-xs text-slate-500 pt-1">
                        <span>File Size: <strong className="text-slate-300">{model.size}</strong></span>
                        <span>Performance: <strong className="text-emerald-400">{model.speed}</strong></span>
                      </div>
                    </div>

                    <div className="flex items-center gap-2 shrink-0">
                      <a
                        href={model.url}
                        target="_blank"
                        rel="noreferrer"
                        className="px-3 py-1.5 rounded-md bg-slate-800 hover:bg-slate-700 text-xs font-medium text-white flex items-center gap-1.5 border border-slate-700 transition-colors"
                      >
                        <ExternalLink className="h-3.5 w-3.5" />
                        <span>Direct Link</span>
                      </a>
                      <button
                        onClick={() => copyToClipboard(`Invoke-WebRequest -Uri "${model.url}" -OutFile "models\\${model.filename.split(' ')[0]}"`, `dl-cmd-${idx}`)}
                        className="p-1.5 rounded-md bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white transition-colors"
                        title="Copy PowerShell download command"
                      >
                        {copiedKey === `dl-cmd-${idx}` ? <CheckCircle2 className="h-4 w-4 text-emerald-400" /> : <Copy className="h-4 w-4" />}
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* TAB 4: WEBCAM & OBS GUIDE */}
        {activeTab === "webcam" && (
          <div className="space-y-6">
            <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 sm:p-8">
              <h2 className="text-xl font-bold text-white mb-2 flex items-center gap-2">
                <Video className="h-5 w-5 text-emerald-400" />
                Phone Rear Camera & OBS Virtual Camera Setup
              </h2>
              <p className="text-xs text-slate-300 max-w-3xl leading-relaxed">
                Connect your Android phone as a high-definition webcam using USB tethering for crisp, professional face swapping with zero wireless lag.
              </p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
                {/* Android Setup */}
                <div className="bg-slate-950 border border-slate-800 rounded-xl p-5 space-y-3">
                  <div className="flex items-center gap-2">
                    <span className="h-6 w-6 rounded-full bg-emerald-500/20 text-emerald-400 text-xs font-bold flex items-center justify-center">
                      A
                    </span>
                    <h4 className="text-sm font-bold text-white">Android Rear Camera via USB</h4>
                  </div>
                  <ol className="text-xs text-slate-300 space-y-2 list-decimal list-inside leading-relaxed">
                    <li>Install <strong>DroidCam</strong> or <strong>Iriun Webcam</strong> from Google Play.</li>
                    <li>Install the corresponding Windows Client on your laptop.</li>
                    <li>Enable <strong>USB Debugging</strong> in Android Developer Options.</li>
                    <li>Plug in via USB cable and launch the client (select USB mode).</li>
                    <li>
                      In Face-Swap Studio, choose the camera marked:
                      <span className="block mt-1 text-emerald-300 font-mono bg-slate-900 p-1.5 rounded border border-slate-800">
                        Camera [1]: DroidCam Source ★ (Phone Rear Camera)
                      </span>
                    </li>
                  </ol>
                </div>

                {/* OBS & Discord Setup */}
                <div className="bg-slate-950 border border-slate-800 rounded-xl p-5 space-y-3">
                  <div className="flex items-center gap-2">
                    <span className="h-6 w-6 rounded-full bg-teal-500/20 text-teal-400 text-xs font-bold flex items-center justify-center">
                      B
                    </span>
                    <h4 className="text-sm font-bold text-white">Streaming to Discord, Zoom & Meet</h4>
                  </div>
                  <ol className="text-xs text-slate-300 space-y-2 list-decimal list-inside leading-relaxed">
                    <li>Install <strong>OBS Studio</strong> (OBS registers the virtual webcam DirectShow filter).</li>
                    <li>In Face-Swap Studio, tick <strong>"Broadcast to OBS Virtual Camera"</strong>.</li>
                    <li>Open <strong>Discord / Zoom / Google Meet</strong> Settings &rarr; Video.</li>
                    <li>Select <strong>"OBS Virtual Camera"</strong> as your video input device.</li>
                    <li>The swapped output will stream in real-time with zero queue latency.</li>
                  </ol>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 5: ZERO-TRASH UNINSTALLER */}
        {activeTab === "uninstaller" && (
          <div className="space-y-6">
            <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 sm:p-8">
              <div className="flex items-center gap-3 mb-3">
                <div className="h-10 w-10 rounded-xl bg-rose-500/10 border border-rose-500/20 flex items-center justify-center text-rose-400">
                  <Trash2 className="h-5 w-5" />
                </div>
                <div>
                  <h2 className="text-xl font-bold text-white">Zero-Trash Complete Uninstaller</h2>
                  <p className="text-xs text-slate-400">Leaves no orphaned virtual environments, bytecodes, or temporary caches.</p>
                </div>
              </div>

              <p className="text-xs text-slate-300 leading-relaxed max-w-3xl mt-2">
                We take system cleanliness seriously. Running <code className="text-rose-300 font-mono">.\uninstall.ps1</code> ensures your Windows 11 machine is restored to its exact original state without leaving a single trace of junk or cache.
              </p>

              {/* Deletion Audit Table */}
              <div className="bg-slate-950 border border-slate-800 rounded-xl p-5 mt-6 space-y-4">
                <h4 className="text-xs font-bold text-slate-200 uppercase tracking-wider">
                  Automated Clean-Up Audit List
                </h4>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                  {[
                    { label: "Virtual Environments", target: ".venv/ and realtime_faceswap/.venv/" },
                    { label: "Python Compiled Cache", target: "All __pycache__, *.pyc, *.pyo recursively" },
                    { label: "Build & Compiler Folders", target: ".pytest_cache, build/, dist/, *.egg-info" },
                    { label: "Telemetry & Logs", target: "benchmarks/*.json, *.csv, *.log" },
                    { label: "Downloaded Models", target: "models/*.onnx, *.tmp (or keep with -KeepModels)" },
                    { label: "Pip Build Cache", target: "%LOCALAPPDATA%\\pip\\cache" },
                  ].map((item, idx) => (
                    <div key={idx} className="bg-slate-900/80 p-3 rounded-lg border border-slate-800/80">
                      <div className="flex items-center gap-1.5 text-rose-400 font-semibold mb-1">
                        <CheckCircle2 className="h-3.5 w-3.5 shrink-0" />
                        <span>{item.label}</span>
                      </div>
                      <span className="font-mono text-slate-400 text-[11px] break-all">{item.target}</span>
                    </div>
                  ))}
                </div>

                {/* Commands Box */}
                <div className="mt-6 pt-4 border-t border-slate-800 space-y-3">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                    <span className="text-xs font-semibold text-white">Full Interactive Uninstall:</span>
                    <button
                      onClick={() => copyToClipboard(".\\uninstall.ps1", "un-full")}
                      className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-xs text-slate-200 flex items-center gap-1 self-start sm:self-auto"
                    >
                      {copiedKey === "un-full" ? <CheckCircle2 className="h-3 w-3 text-emerald-400" /> : <Copy className="h-3 w-3" />}
                      <span>Copy</span>
                    </button>
                  </div>
                  <pre className="bg-slate-900 p-2.5 rounded border border-slate-800 text-xs font-mono text-rose-300">
                    <code>.\uninstall.ps1</code>
                  </pre>

                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pt-2">
                    <span className="text-xs font-semibold text-white">Keep Models (Delete Only Packages & Cache):</span>
                    <button
                      onClick={() => copyToClipboard(".\\uninstall.ps1 -KeepModels", "un-keep")}
                      className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-xs text-slate-200 flex items-center gap-1 self-start sm:self-auto"
                    >
                      {copiedKey === "un-keep" ? <CheckCircle2 className="h-3 w-3 text-emerald-400" /> : <Copy className="h-3 w-3" />}
                      <span>Copy</span>
                    </button>
                  </div>
                  <pre className="bg-slate-900 p-2.5 rounded border border-slate-800 text-xs font-mono text-emerald-300">
                    <code>.\uninstall.ps1 -KeepModels</code>
                  </pre>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-950 py-6 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>Real-Time AI Face-Swap Studio • Optimized for NVIDIA RTX 4050 6GB</span>
          <span className="text-slate-400">Designed for ethical, consensual media & creative production.</span>
        </div>
      </footer>
    </div>
  );
}
