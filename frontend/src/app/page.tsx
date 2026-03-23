"use client";
import dynamic from "next/dynamic";

const DrawingCanvas = dynamic(() => import("@/components/DrawingCanvas"), {
  ssr: false,
});

export default function Home() {
  return (
    <main className="min-h-screen bg-black">
      <DrawingCanvas />
    </main>
  );
}
