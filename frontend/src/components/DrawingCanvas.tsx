'use client';

import React, { useRef, useEffect, useState, useCallback } from 'react';
import type { Results } from '@mediapipe/hands';

declare global {
  interface Window {
    Hands: any;
    Camera: any;    
  }
}

interface PredictionData {
  prediction: string;
  confidence: number;
}

interface SegmentResult {
  predictions: string[];
}

export default function DrawingCanvas() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRefView = useRef<HTMLCanvasElement>(null); // To see the drawing mirrored
  const canvasRefClean = useRef<HTMLCanvasElement>(null); // To send for prediction (white on black)

  const [isDrawing, setIsDrawing] = useState(false);
  const [isCameraReady, setIsCameraReady] = useState(false);
  const [predictions, setPredictions] = useState<PredictionData[]>([]);
  const [history, setHistory] = useState<string[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isError, setIsError] = useState(false);

  const currentPath = useRef<{ xValueScreen: number, xValueModel: number, yValue: number }[]>([]);
  const stopTimeout = useRef<NodeJS.Timeout | null>(null);
  const mediaPipeRefs = useRef<{ handsModel: any, camera: any }>({ handsModel: null, camera: null });

  const clearCanvases = useCallback(() => {
    if (canvasRefView.current) {
      const ctx = canvasRefView.current.getContext('2d');
      if (ctx) ctx.clearRect(0, 0, canvasRefView.current.width, canvasRefView.current.height);
    }
    if (canvasRefClean.current) {
      const ctx = canvasRefClean.current.getContext('2d');
      if (ctx) {
        // Fill black so next export is opaque, not transparent
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, canvasRefClean.current.width, canvasRefClean.current.height);
        // Then clear back to transparent for fresh drawing
        ctx.clearRect(0, 0, canvasRefClean.current.width, canvasRefClean.current.height);
      }
    }
  }, []);

  const resetAll = useCallback(() => {
    clearCanvases();
    setPredictions([]);
    setHistory([]);
    setIsError(false);
  }, [clearCanvases]);

  const speakPrediction = useCallback((text: string) => {
    if ('speechSynthesis' in window) {
      const msg = new SpeechSynthesisUtterance(`I see ${text}`);
      window.speechSynthesis.speak(msg);
    }
  }, []);

  const predictDrawingSegment = useCallback(async () => {
    if (!canvasRefClean.current) return;
    setIsProcessing(true);
    setIsError(false);

    const canvas = canvasRefClean.current;

    // Check if anything is actually on the canvas
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.toBlob(async (blob) => {
      if (!blob) return;
      const formData = new FormData();
      formData.append('file', blob, 'segment.png');

      try {
        const res = await fetch('http://127.0.0.1:8000/predict', {
          method: 'POST',
          body: formData,
        });

        if (!res.ok) {
          const body = await res.text();
          console.warn("Predict request failed:", res.status, body);
          setIsError(true);
          setPredictions([]);
          return;
        }

        const data = await res.json();

        if (data.prediction) {
          const predVal = data.prediction.replace('Num_', '').replace('Shape_', 'shape ');
          const newPred = { prediction: predVal, confidence: data.confidence };
          setPredictions(prev => [newPred, ...prev]);
          setHistory(prev => [predVal, ...prev].slice(0, 10));
          speakPrediction(predVal);
          setIsError(false);
        } else {
          console.warn("No valid prediction returned", data);
          setIsError(true);
          setPredictions([]);
        }
      } catch (err) {
        console.warn("Prediction error:", err);
        setIsError(true);
        setPredictions([]);
      } finally {
        setIsProcessing(false);
        // Clear both canvases for the next segment
        clearCanvases();
      }
    });
  }, [clearCanvases, speakPrediction]);

  const onResults = useCallback((results: Results) => {
    if (!canvasRefView.current || !canvasRefClean.current || !videoRef.current) return;

    const canvasV = canvasRefView.current;
    const canvasC = canvasRefClean.current;
    const ctxV = canvasV.getContext('2d');
    const ctxC = canvasC.getContext('2d');
    if (!ctxV || !ctxC) return;

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      const landmarks = results.multiHandLandmarks[0];
      const indexFinger = landmarks[8];

      // More robust palm open detection: thumb-pinky distance vs index-palm distance
      const thumbTip = landmarks[4];
      const pinkyTip = landmarks[20];
      const distTP = Math.sqrt(Math.pow(thumbTip.x - pinkyTip.x, 2) + Math.pow(thumbTip.y - pinkyTip.y, 2));
      const palmOpen = distTP > 0.15; // roughly empirical

      const xView = indexFinger.x * canvasV.width;
      const xModel = (1 - indexFinger.x) * canvasC.width; // Flip horizontally for prediction
      const y = indexFinger.y * canvasV.height;

      setIsDrawing(true);
      const strokeWidth = 20; // Increased from 14 for better model recognition

      if (currentPath.current.length > 0) {
        const prevItem = currentPath.current[currentPath.current.length - 1];
        const prevXView = prevItem.xValueScreen;
        const prevXModel = prevItem.xValueModel;
        const prevY = prevItem.yValue;

        // Draw on view canvas (colored/stylized) - this is mirrored via CSS
        ctxV.beginPath();
        ctxV.moveTo(prevXView, prevY);
        ctxV.lineTo(xView, y);
        ctxV.strokeStyle = '#3b82f6';
        ctxV.lineWidth = strokeWidth;
        ctxV.lineCap = 'round';
        ctxV.shadowBlur = 10;
        ctxV.shadowColor = '#3b82f6';
        ctxV.stroke();

        // Draw on clean prediction canvas (black bg, white strokes for model)
        // Fill black background on first stroke point so PNG is not transparent
        if (currentPath.current.length === 1) {
          ctxC.fillStyle = 'black';
          ctxC.fillRect(0, 0, canvasC.width, canvasC.height);
        }
        ctxC.beginPath();
        ctxC.moveTo(prevXModel, prevY);
        ctxC.lineTo(xModel, y);
        ctxC.strokeStyle = 'white';
        ctxC.lineWidth = strokeWidth;
        ctxC.lineCap = 'round';
        ctxC.shadowBlur = 0;
        ctxC.stroke();
      }

      currentPath.current.push({
        xValueScreen: xView,
        xValueModel: xModel,
        yValue: y
      });

      if (stopTimeout.current) clearTimeout(stopTimeout.current);

      // Stop detection: no movement for 1.5s or open palm
      const timeoutVal = palmOpen ? 500 : 1500;
      stopTimeout.current = setTimeout(() => {
        setIsDrawing(false);
        if (currentPath.current.length > 5) {
          predictDrawingSegment();
        }
        currentPath.current = [];
      }, timeoutVal);
    }
  }, [clearCanvases, predictDrawingSegment]);

  useEffect(() => {
    let isComponentMounted = true;
    const initMP = () => {
      if (!isComponentMounted || !window.Hands || !window.Camera) {
        if (isComponentMounted) setTimeout(initMP, 500);
        return;
      }
      const handsModel = new window.Hands({
        locateFile: (file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
      });
      handsModel.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.7,
        minTrackingConfidence: 0.7
      });
      handsModel.onResults(onResults);
      mediaPipeRefs.current.handsModel = handsModel;

      if (videoRef.current) {
        const camera = new window.Camera(videoRef.current, {
          onFrame: async () => {
            if (videoRef.current && isComponentMounted) {
              await handsModel.send({ image: videoRef.current });
            }
          },
          width: 640,
          height: 480
        });
        mediaPipeRefs.current.camera = camera;
        camera.start().then(() => { if (isComponentMounted) setIsCameraReady(true); });
      }
    };
    initMP();
    return () => { isComponentMounted = false; };
  }, [onResults]);

  return (
    <div className="flex flex-col h-screen w-full bg-[#050505] text-white overflow-hidden">
      {/* Header bar */}
      <header className="h-16 px-8 flex items-center justify-between border-b border-gray-900 bg-black/50 backdrop-blur-md z-30">
        <div className="flex items-center gap-4">
          <div className="w-8 h-8 rounded-full bg-blue-500 animate-pulse ring-4 ring-blue-500/20" />
          <h1 className="text-xl font-bold tracking-tight">AI Air Drawing <span className="text-gray-500 font-normal">v2 Dual-View</span></h1>
        </div>

        <div className="flex items-center gap-6">
          <div className="flex flex-col items-end">
            <span className="text-xs text-gray-500 uppercase tracking-widest font-semibold">Latest Prediction</span>
            <span className="text-blue-400 font-mono text-lg font-bold">
              {predictions[0] ? `${predictions[0].prediction} (${(predictions[0].confidence * 100).toFixed(0)}%)` : "---"}
            </span>
          </div>
          <button
            onClick={resetAll}
            className="px-5 py-2 secondary-btn rounded-full bg-gray-900 border border-gray-800 hover:border-blue-500/50 transition-all active:scale-95"
          >
            Reset All
          </button>
        </div>
      </header>

      {/* Main split view */}
      <main className="flex-1 flex w-full relative">
        {/* Left: Clean Canvas view */}
        <section className="flex-1 border-r border-gray-900 relative flex flex-col group">
          <div className="absolute top-4 left-6 z-20 pointer-events-none">
            <span className="px-3 py-1 rounded bg-black/40 border border-white/5 text-[10px] uppercase tracking-tighter text-gray-400">Virtual Output</span>
          </div>
          <div className="flex-1 relative overflow-hidden bg-black flex items-center justify-center">
            <canvas
              ref={canvasRefView}
              width={640}
              height={480}
              className="relative z-10 w-full max-w-2xl h-auto aspect-[4/3] transform -scale-x-100"
            />
            <canvas ref={canvasRefClean} width={640} height={480} className="hidden" />
          </div>
          <div className="p-4 border-t border-gray-900 flex gap-2 overflow-x-auto no-scrollbar bg-black/20 h-24 items-center">
            {history.map((h, i) => (
              <div key={i} className="flex-shrink-0 px-4 py-2 rounded-lg bg-gray-900 border border-gray-800 text-sm font-medium animate-in fade-in slide-in-from-left-4 capitalize">
                {h}
              </div>
            ))}
            {history.length === 0 && <span className="text-gray-700 text-sm italic mx-auto">No history yet... start drawing in the air</span>}
          </div>
        </section>

        {/* Right: Camera view */}
        <section className="flex-1 relative flex flex-col">
          <div className="absolute top-4 right-6 z-20 pointer-events-none">
            <span className="px-3 py-1 rounded bg-blue-500/20 border border-blue-500/30 text-[10px] uppercase tracking-tighter text-blue-400">Tracking Engine</span>
          </div>
          <div className="flex-1 relative bg-black flex items-center justify-center overflow-hidden">
            <video
              ref={videoRef}
              className="absolute inset-0 w-full h-full object-cover opacity-100 transform -scale-x-100"
              autoPlay
              playsInline
            />
            {!isCameraReady && (
              <div className="absolute inset-0 bg-black z-10 flex items-center justify-center">
                <div className="flex flex-col items-center gap-4">
                  <div className="w-12 h-12 border-4 border-blue-500/30 border-t-blue-500 rounded-full animate-spin" />
                  <span className="text-gray-500 font-medium">Initializing camera...</span>
                </div>
              </div>
            )}
          </div>
          <div className="p-6 bg-black border-t border-gray-900">
            <div className="flex gap-4 items-center mb-4">
              <div className={`w-2 h-2 rounded-full ${isDrawing ? 'bg-green-500 animate-ping' : 'bg-red-500'}`} />
              <span className="text-sm font-semibold tracking-wide uppercase">{isDrawing ? 'Recording Strokes...' : 'Waiting for hand...'}</span>
              {isProcessing && <span className="ml-auto text-xs text-blue-400 animate-pulse font-mono tracking-tighter italic">Predicting...</span>}
              {isError && <span className="ml-auto text-xs text-red-500 font-mono tracking-tighter italic">Server offline / Error</span>}
            </div>
            <div className="w-full bg-gray-900 h-1 rounded-full overflow-hidden">
              <div className={`h-full bg-blue-500 transition-all duration-300 ${isDrawing ? 'w-full opa-100' : 'w-0 opa-0'}`} />
            </div>
          </div>
        </section>
      </main>

      {/* Floating Result Panel */}
      <div className="absolute bottom-10 left-1/2 -translate-x-1/2 flex flex-col gap-3 min-w-[300px] z-40">
        {isError && (
          <div className="bg-red-950/90 text-red-300 border border-red-600 p-4 rounded-2xl text-sm font-semibold">
            Prediction failed. Ensure backend is running and model is loaded.
          </div>
        )}
        {!isError && predictions.length === 0 && (
          <div className="bg-gray-900/80 text-gray-300 border border-gray-700 p-4 rounded-2xl text-sm font-medium">
            No predictions yet — draw in the air and hold still for a moment.
          </div>
        )}
        {predictions.slice(0, 1).map((pred, i) => (
          <div key={i} className="bg-gray-900/80 backdrop-blur-xl p-6 rounded-3xl border border-white/10 shadow-2xl animate-in zoom-in-95 duration-200">
            <div className="flex items-end justify-between gap-12 mb-4">
              <span className="text-4xl font-black capitalize tracking-tight text-blue-500">{pred.prediction}</span>
              <span className="text-sm font-mono text-gray-500">{(pred.confidence * 100).toFixed(1)}% match</span>
            </div>
            <div className="w-full h-1.5 bg-black/40 rounded-full overflow-hidden mb-2">
              <div
                className="h-full bg-gradient-to-r from-blue-600 to-indigo-500 transition-all duration-1000"
                style={{ width: `${pred.confidence * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
