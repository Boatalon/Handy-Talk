'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { FiX, FiCamera, FiCameraOff, FiVolume2 } from 'react-icons/fi';

// ---- TGCN constants (mirrors tgcn-input.js) ----
const TGCN_FRAME_COUNT = 50;
const TGCN_NODE_COUNT = 55;
const TGCN_FEATURES_PER_NODE = TGCN_FRAME_COUNT * 2; // x,y per frame

const MISSING: [number, number] = [-1, -1];

function norm(lm: any): [number, number] {
    if (!lm || !isFinite(lm.x) || !isFinite(lm.y)) return [...MISSING];
    return [2 * lm.x - 1, 2 * lm.y - 1];
}

function mid(a: any, b: any): [number, number] {
    if (!a || !b || !isFinite(a.x) || !isFinite(b.x)) return [...MISSING];
    const na = norm(a), nb = norm(b);
    return [(na[0] + nb[0]) / 2, (na[1] + nb[1]) / 2];
}

function mapHand(lms: any[]): [number, number][] {
    return Array.from({ length: 21 }, (_, i) => norm(lms?.[i]));
}

// HAND_CONNECTIONS for skeleton overlay
const HAND_CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [0,9],[9,10],[10,11],[11,12],
    [0,13],[13,14],[14,15],[15,16],
    [0,17],[17,18],[18,19],[19,20],
    [5,9],[9,13],[13,17],
];

function holisticToTgcnFrame(result: any): [number, number][] | null {
    // Tasks-Vision HolisticLandmarker: poseLandmarks is NormalizedLandmark[][]
    const pose = result?.poseLandmarks?.[0];
    if (!pose || pose.length < 25) return null;

    const body: [number, number][] = [
        norm(pose[0]),
        mid(pose[11], pose[12]),
        norm(pose[12]),
        norm(pose[14]),
        norm(pose[16]),
        norm(pose[11]),
        norm(pose[13]),
        norm(pose[15]),
        mid(pose[23], pose[24]),
        norm(pose[5]),
        norm(pose[2]),
        norm(pose[8]),
        norm(pose[7]),
    ];

    const left = mapHand(result?.leftHandLandmarks?.[0] ?? []);
    const right = mapHand(result?.rightHandLandmarks?.[0] ?? []);
    return [...body, ...left, ...right];
}

function buildFloat32(frames: [number, number][][]): Float32Array {
    const out = new Float32Array(TGCN_NODE_COUNT * TGCN_FEATURES_PER_NODE);
    for (let n = 0; n < TGCN_NODE_COUNT; n++) {
        const offset = n * TGCN_FEATURES_PER_NODE;
        for (let f = 0; f < TGCN_FRAME_COUNT; f++) {
            const [x, y] = frames[f][n];
            out[offset + f * 2] = x;
            out[offset + f * 2 + 1] = y;
        }
    }
    return out;
}

function softmax(arr: Float32Array): number[] {
    let max = -Infinity;
    for (const v of arr) if (v > max) max = v;
    const exps = Array.from(arr).map(v => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(v => v / sum);
}

// Timing constants: 25 FPS frame collection
const TARGET_FPS = 25;
const FRAME_INTERVAL_MS = 1000 / TARGET_FPS; // 40ms

interface PredictionResult { word: string; probability: number; }
interface HandKeypointDemoProps { isOpen: boolean; onClose: () => void; }

const HandKeypointDemo = ({ isOpen, onClose }: HandKeypointDemoProps) => {
    const [mounted, setMounted] = useState(false);
    useEffect(() => { setMounted(true); }, []);

    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const holisticRef = useRef<any>(null);
    const sessionRef = useRef<any>(null);
    const labelsRef = useRef<string[]>([]);
    const frameBufferRef = useRef<[number, number][][]>([]);
    const animRef = useRef<number>(0);
    const lastSpokenRef = useRef<{ word: string; time: number }>({ word: '', time: 0 });
    const isPredictingRef = useRef(false);
    const streamRef = useRef<MediaStream | null>(null);
    const lastInferenceTimeRef = useRef<number>(0);
    const lastFrameTimeRef = useRef<number>(0);
    const framesSinceLastInferenceRef = useRef<number>(0);
    const predictionHistoryRef = useRef<string[]>([]);
    
    // Configuration constants
    const INFERENCE_STRIDE = 5;
    const CONFIDENCE_THRESHOLD = 0.6;
    const VOTE_WINDOW = 5;
    const REQUIRED_VOTES = 3;

    const [frameCount, setFrameCount] = useState(0);
    const [cooldownRemaining, setCooldownRemaining] = useState(0);
    const [isLoading, setIsLoading] = useState(true);
    const [loadingMsg, setLoadingMsg] = useState('Initializing…');
    const [error, setError] = useState<string | null>(null);
    const [predictions, setPredictions] = useState<PredictionResult[]>([]);
    const [poseVisible, setPoseVisible] = useState(false);

    const drawSkeleton = useCallback((result: any) => {
        const canvas = canvasRef.current;
        const video = videoRef.current;
        if (!canvas || !video) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Tasks-Vision: poseLandmarks is NormalizedLandmark[][] — index [0] for first person
        const pose = result?.poseLandmarks?.[0];
        const hasPose = !!(pose && pose.length > 0);
        setPoseVisible(hasPose);

        if (!hasPose) return;

        const W = canvas.width, H = canvas.height;

        // Upper-body pose connections
        const poseConnections = [
            [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
            [11, 23], [12, 24], [23, 24], [0, 11], [0, 12],
        ];
        ctx.strokeStyle = 'rgba(255,165,0,0.8)';
        ctx.lineWidth = 2;
        for (const [s, e] of poseConnections) {
            const a = pose[s], b = pose[e];
            if (!a || !b) continue;
            ctx.beginPath();
            ctx.moveTo(a.x * W, a.y * H);
            ctx.lineTo(b.x * W, b.y * H);
            ctx.stroke();
        }
        // Pose dots
        for (const lm of pose.slice(0, 25)) {
            if (!lm) continue;
            ctx.beginPath();
            ctx.arc(lm.x * W, lm.y * H, 4, 0, Math.PI * 2);
            ctx.fillStyle = '#ffb347';
            ctx.shadowColor = 'rgba(255,165,0,0.8)';
            ctx.shadowBlur = 8;
            ctx.fill();
            ctx.shadowBlur = 0;
        }

        // Hand skeleton — leftHandLandmarks[0] and rightHandLandmarks[0]
        for (const rawHand of [result?.leftHandLandmarks?.[0], result?.rightHandLandmarks?.[0]]) {
            if (!rawHand || rawHand.length < 21) continue;
            // Draw connections
            ctx.strokeStyle = 'rgba(255,165,0,0.7)';
            ctx.lineWidth = 2;
            for (const [s, e] of HAND_CONNECTIONS) {
                const a = rawHand[s], b = rawHand[e];
                if (!a || !b) continue;
                ctx.beginPath();
                ctx.moveTo(a.x * W, a.y * H);
                ctx.lineTo(b.x * W, b.y * H);
                ctx.stroke();
            }
            // Draw dots
            rawHand.forEach((lm: any, i: number) => {
                if (!lm) return;
                const isTip = [4, 8, 12, 16, 20].includes(i);
                ctx.beginPath();
                ctx.arc(lm.x * W, lm.y * H, isTip ? 7 : 4, 0, Math.PI * 2);
                ctx.fillStyle = isTip ? '#ff8c00' : '#ffb347';
                ctx.strokeStyle = 'white';
                ctx.lineWidth = 1.5;
                ctx.shadowColor = 'rgba(255,165,0,0.8)';
                ctx.shadowBlur = isTip ? 12 : 6;
                ctx.fill();
                ctx.stroke();
                ctx.shadowBlur = 0;
            });
        }
    }, []);

    const runInference = useCallback(async () => {
        if (isPredictingRef.current) return;
        if (frameBufferRef.current.length < TGCN_FRAME_COUNT) return;
        if (!sessionRef.current || labelsRef.current.length === 0) return;

        try {
            isPredictingRef.current = true;
            const input = buildFloat32(frameBufferRef.current.slice(-TGCN_FRAME_COUNT));
            const ort = await import('onnxruntime-web');
            const tensor = new ort.Tensor('float32', input, [1, TGCN_NODE_COUNT, TGCN_FEATURES_PER_NODE]);
            const outputs = await sessionRef.current.run({ keypoints: tensor });
            const probs = softmax(outputs.logits.data as Float32Array);
            const top5 = probs
                .map((p, i) => ({ word: labelsRef.current[i], probability: p }))
                .sort((a, b) => b.probability - a.probability)
                .slice(0, 5);

            setPredictions(top5);
            lastInferenceTimeRef.current = Date.now();

            // Voting system
            const top = top5[0];
            let currentWord = "none";
            if (top && top.probability >= CONFIDENCE_THRESHOLD) {
                currentWord = top.word;
            }

            predictionHistoryRef.current.push(currentWord);
            if (predictionHistoryRef.current.length > VOTE_WINDOW) {
                predictionHistoryRef.current.shift();
            }

            const counts = predictionHistoryRef.current.reduce((acc, w) => {
                acc[w] = (acc[w] || 0) + 1;
                return acc;
            }, {} as Record<string, number>);

            let bestVoteWord = "none";
            let maxVotes = 0;
            for (const [w, count] of Object.entries(counts)) {
                if (count > maxVotes) {
                    maxVotes = count;
                    bestVoteWord = w;
                }
            }

            if (bestVoteWord !== "none" && maxVotes >= REQUIRED_VOTES) {
                if (bestVoteWord !== lastSpokenRef.current.word || (Date.now() - lastSpokenRef.current.time > 4000)) {
                    if (!window.speechSynthesis.speaking) {
                        const utt = new SpeechSynthesisUtterance(bestVoteWord);
                        utt.lang = 'en-US';
                        window.speechSynthesis.speak(utt);
                        lastSpokenRef.current = { word: bestVoteWord, time: Date.now() };
                    }
                }
            }
        } catch (e) {
            console.error('Inference error:', e);
        } finally {
            isPredictingRef.current = false;
        }
    }, []);

    const processFrameRef = useRef<() => void>(() => {});
    const lastVideoTimeRef = useRef<number>(-1);

    processFrameRef.current = async () => {
        const video = videoRef.current;
        const holistic = holisticRef.current;
        
        if (!video || !holistic || video.readyState < 2) {
            animRef.current = requestAnimationFrame(processFrameRef.current);
            return;
        }

        if (video.currentTime !== lastVideoTimeRef.current) {
            lastVideoTimeRef.current = video.currentTime;
            try {
                const now = performance.now();
                const result = holistic.detectForVideo(video, now);
                drawSkeleton(result);

                setCooldownRemaining(0);
                
                // VAD: Check if hands are present in this frame
                const hasHands = (result?.leftHandLandmarks?.[0]?.length > 0) || (result?.rightHandLandmarks?.[0]?.length > 0);

                // Collect frames at 25 FPS (every 40ms)
                if (now - lastFrameTimeRef.current >= FRAME_INTERVAL_MS) {
                    lastFrameTimeRef.current = now;

                    const frame = holisticToTgcnFrame(result);
                    if (frame) {
                        frameBufferRef.current.push(frame);
                        if (frameBufferRef.current.length > TGCN_FRAME_COUNT) frameBufferRef.current.shift();
                        setFrameCount(frameBufferRef.current.length);
                    }

                    if (frameBufferRef.current.length === TGCN_FRAME_COUNT) {
                        framesSinceLastInferenceRef.current++;
                        
                        if (framesSinceLastInferenceRef.current >= INFERENCE_STRIDE) {
                            framesSinceLastInferenceRef.current = 0;
                            
                            if (hasHands) {
                                runInference();
                            } else {
                                predictionHistoryRef.current.push("none");
                                if (predictionHistoryRef.current.length > VOTE_WINDOW) {
                                    predictionHistoryRef.current.shift();
                                }
                            }
                        }
                    }
                }
            } catch (err: any) {
                console.error("MediaPipe detection error:", err);
                setError(err?.message || "MediaPipe detection failed.");
            }
        }
        animRef.current = requestAnimationFrame(processFrameRef.current);
    };

    const initialize = useCallback(async () => {
        try {
            setIsLoading(true);
            setError(null);
            frameBufferRef.current = [];
            setFrameCount(0);

            setLoadingMsg('Loading MediaPipe Holistic…');
            const vision = await import('@mediapipe/tasks-vision');
            const filesetResolver = await vision.FilesetResolver.forVisionTasks(
                'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
            );
            const holistic = await vision.HolisticLandmarker.createFromOptions(filesetResolver, {
                baseOptions: {
                    modelAssetPath:
                        'https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task',
                    delegate: 'CPU',
                },
                runningMode: 'VIDEO',
                minPoseDetectionConfidence: 0.5,
                minPosePresenceConfidence: 0.5,
                outputFaceBlendshapes: false,
            });
            holisticRef.current = holistic;

            setLoadingMsg('Loading TGCN ONNX model (~3.5 MB)…');
            const ort = await import('onnxruntime-web');
            ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
            const session = await ort.InferenceSession.create('/models/tgcn_wlasl100.onnx', {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all',
            });
            sessionRef.current = session;

            setLoadingMsg('Loading labels…');
            const res = await fetch('/models/labels.json');
            labelsRef.current = await res.json();

            setLoadingMsg('Starting camera…');
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480, facingMode: 'user' },
                audio: false,
            });
            streamRef.current = stream;
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                await videoRef.current.play();
            }

            setIsLoading(false);
            animRef.current = requestAnimationFrame(processFrameRef.current);
        } catch (err: any) {
            console.error('Init error:', err);
            setError(err?.name === 'NotAllowedError'
                ? 'Camera permission denied. Allow access in browser settings.'
                : err?.message || 'Initialization failed. Please try again.');
            setIsLoading(false);
        }
    }, []); // eslint-disable-line react-hooks/exhaustive-deps

    useEffect(() => {
        if (isOpen) initialize();
    }, [isOpen]); // eslint-disable-line react-hooks/exhaustive-deps

    useEffect(() => {
        if (!isOpen) {
            cancelAnimationFrame(animRef.current);
            streamRef.current?.getTracks().forEach(t => t.stop());
            streamRef.current = null;
            holisticRef.current?.close?.();
            holisticRef.current = null;
            sessionRef.current = null;
            frameBufferRef.current = [];
            lastInferenceTimeRef.current = 0;
            lastFrameTimeRef.current = 0;
            framesSinceLastInferenceRef.current = 0;
            predictionHistoryRef.current = [];
            setCooldownRemaining(0);
            setIsLoading(true);
            setError(null);
            setPredictions([]);
            setPoseVisible(false);
            setFrameCount(0);
            window.speechSynthesis.cancel();
        }
    }, [isOpen]);

    useEffect(() => {
        if (!isOpen) return;
        const handle = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
        document.body.style.overflow = 'hidden';
        document.addEventListener('keydown', handle);
        return () => { document.body.style.overflow = ''; document.removeEventListener('keydown', handle); };
    }, [isOpen, onClose]);

    if (!mounted) return null;

    return createPortal(
        <AnimatePresence>
            {isOpen && (
                <motion.div
                    initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                    className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-sm"
                    onClick={e => { if (e.target === e.currentTarget) onClose(); }}
                >
                    <motion.div
                        role="dialog" aria-modal="true" aria-labelledby="asl-demo-title"
                        initial={{ scale: 0.9, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.9, opacity: 0 }}
                        transition={{ type: 'spring', damping: 25, stiffness: 300 }}
                        className="relative w-[95vw] max-w-6xl h-[90vh] md:h-[82vh] min-h-[500px] bg-gray-900 rounded-2xl overflow-hidden shadow-2xl flex flex-col md:flex-row"
                    >
                        {/* Camera */}
                        <div className="relative flex-1 bg-black flex flex-col min-h-0">
                            <div className="absolute top-0 left-0 right-0 z-10 flex items-center justify-between px-4 py-3 bg-gradient-to-b from-black/80 to-transparent">
                                <div className="flex items-center gap-2">
                                    <FiCamera className="w-5 h-5 text-white" />
                                    <h3 id="asl-demo-title" className="text-sm font-bold text-white tracking-wide uppercase">
                                        Live ASL — WLASL100 · Pose-TGCN
                                    </h3>
                                </div>
                                <button onClick={onClose} aria-label="Close" className="md:hidden w-10 h-10 rounded-full bg-white/20 flex items-center justify-center">
                                    <FiX className="w-5 h-5 text-white" />
                                </button>
                            </div>

                            <div className="relative flex-1">
                                <video ref={videoRef} className="absolute inset-0 w-full h-full object-cover" playsInline muted style={{ transform: 'scaleX(-1)' }} />
                                <canvas ref={canvasRef} className="absolute inset-0 w-full h-full object-cover" style={{ transform: 'scaleX(-1)' }} />

                                {isLoading && (
                                    <div role="status" className="absolute inset-0 flex flex-col items-center justify-center bg-gray-900/90 z-20">
                                        <div className="w-12 h-12 border-4 border-amber-500 border-t-transparent rounded-full animate-spin mb-4" />
                                        <p className="text-amber-500 font-bold text-lg">Initializing</p>
                                        <p className="text-gray-300 text-sm mt-2">{loadingMsg}</p>
                                    </div>
                                )}
                                {error && (
                                    <div role="alert" className="absolute inset-0 flex flex-col items-center justify-center bg-gray-900/90 z-20">
                                        <FiCameraOff className="w-12 h-12 text-red-500 mb-4" />
                                        <p className="text-white text-lg font-bold mb-2">Error</p>
                                        <p className="text-gray-400 text-sm text-center max-w-sm px-4">{error}</p>
                                        <button onClick={initialize} className="mt-6 px-6 py-2 bg-amber-600 text-white rounded-lg hover:bg-amber-700 transition font-medium">
                                            Try Again
                                        </button>
                                    </div>
                                )}
                            </div>

                            <div className="absolute bottom-0 left-0 right-0 z-10 p-3 bg-gradient-to-t from-black/90 to-transparent flex items-center gap-3">
                                <div className={`w-3 h-3 rounded-full ${
                                    cooldownRemaining > 0
                                        ? 'bg-amber-500 animate-pulse'
                                        : poseVisible
                                            ? 'bg-green-500 animate-pulse'
                                            : 'bg-red-500'
                                }`} />
                                <span className="text-white text-sm font-medium">
                                    {cooldownRemaining > 0
                                        ? `Delay: ${cooldownRemaining}s — เตรียมท่าถัดไป`
                                        : poseVisible
                                            ? frameCount === TGCN_FRAME_COUNT
                                                ? 'Translating...'
                                                : `Recording sign: ${frameCount}/${TGCN_FRAME_COUNT} frames (~2s)`
                                            : 'Show upper body + hands'}
                                </span>
                                {cooldownRemaining === 0 && (
                                    <span className="ml-auto text-gray-400 text-xs tabular-nums">
                                        {frameCount}/{TGCN_FRAME_COUNT} frames
                                    </span>
                                )}
                            </div>
                        </div>

                        {/* Results panel */}
                        <div className="w-full md:w-80 bg-gray-900 border-l border-gray-800 flex flex-col relative z-20">
                            <button onClick={onClose} aria-label="Close"
                                className="absolute top-4 right-4 hidden md:flex w-8 h-8 rounded-full bg-gray-800 hover:bg-gray-700 items-center justify-center transition-colors z-30 text-gray-400 hover:text-white"
                            >
                                <FiX className="w-5 h-5" />
                            </button>

                            <div className="p-6 border-b border-gray-800 mt-2">
                                <h3 className="text-amber-500 font-bold text-lg mb-1">Translation Results</h3>
                                <p className="text-gray-400 text-xs leading-relaxed">
                                    Pose-TGCN · WLASL100 (100 words)<br />
                                    Top-1 ≈ 55% · Top-5 ≈ 79%<br />
                                    Perform one word over ~2 seconds.
                                </p>
                            </div>

                            <div className="flex-1 overflow-y-auto p-6">
                                {predictions.length > 0 ? (
                                    <div className="space-y-3">
                                        <AnimatePresence>
                                            {predictions.map((pred, i) => {
                                                const pct = (pred.probability * 100).toFixed(1);
                                                const isTop = i === 0;
                                                return (
                                                    <motion.div key={pred.word} layout
                                                        initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }}
                                                        className={`relative overflow-hidden rounded-xl border ${isTop ? 'border-amber-500/50 bg-amber-500/10' : 'border-gray-800 bg-gray-800/50'} p-3`}
                                                    >
                                                        <motion.div
                                                            className={`absolute left-0 top-0 bottom-0 ${isTop ? 'bg-amber-500/20' : 'bg-gray-700/50'} z-0`}
                                                            initial={{ width: 0 }}
                                                            animate={{ width: `${Math.max(4, pred.probability * 100)}%` }}
                                                            transition={{ duration: 0.3 }}
                                                        />
                                                        <div className="relative z-10 flex items-center justify-between">
                                                            <div className="flex items-center gap-2">
                                                                <span className={`text-lg font-bold capitalize ${isTop ? 'text-amber-400' : 'text-gray-300'}`}>{pred.word}</span>
                                                                {isTop && pred.probability >= 0.4 && <FiVolume2 className="w-4 h-4 text-amber-500/70" />}
                                                            </div>
                                                            <span className={`text-sm font-mono ${isTop ? 'text-amber-400' : 'text-gray-400'}`}>{pct}%</span>
                                                        </div>
                                                    </motion.div>
                                                );
                                            })}
                                        </AnimatePresence>
                                    </div>
                                ) : (
                                    <div className="h-full flex flex-col items-center justify-center opacity-50 space-y-4">
                                        <div className="w-16 h-16 rounded-full border-2 border-dashed border-gray-600 flex items-center justify-center">
                                            <div className="w-2 h-2 rounded-full bg-gray-600" />
                                        </div>
                                        <p className="text-gray-400 text-sm text-center">
                                            Collecting {TGCN_FRAME_COUNT} frames…<br />Show upper body + hands
                                        </p>
                                    </div>
                                )}
                            </div>
                        </div>
                    </motion.div>
                </motion.div>
            )}
        </AnimatePresence>,
        document.body
    );
};

export default HandKeypointDemo;
