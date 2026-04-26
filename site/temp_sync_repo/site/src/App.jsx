import React, { useState, useEffect, useRef } from 'react';
import './index.css';
import { Header } from './components/Header';
import { StatsRow } from './components/StatsRow';
import { Controls } from './components/Controls';
import { OrbitalCanvas } from './components/OrbitalCanvas';
import { RewardChart } from './components/RewardChart';
import { ReasoningPanel } from './components/ReasoningPanel';

function App() {
  const [data, setData] = useState(null);
  const [episodeIdx, setEpisodeIdx] = useState(-1);
  const [stepIdx, setStepIdx] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(8);
  const [status, setStatus] = useState({ type: 'idle', text: 'IDLE' });

  const timerRef = useRef(null);
  const transitionTimerRef = useRef(null);

  // Fascinating Milky Way Cursor Effect
  useEffect(() => {
    let lastTime = 0;
    const handleMouseMove = (e) => {
      const now = Date.now();
      if (now - lastTime < 50) return;
      lastTime = now;
      const numSparkles = Math.floor(Math.random() * 2) + 1;
      for (let i = 0; i < numSparkles; i++) {
        const sparkle = document.createElement('div');
        sparkle.className = 'sparkle';
        const offsetX = (Math.random() - 0.5) * 35;
        const offsetY = (Math.random() - 0.5) * 35;
        sparkle.style.left = `${e.clientX + offsetX}px`;
        sparkle.style.top = `${e.clientY + offsetY}px`;
        sparkle.style.backgroundColor = 'rgba(255, 255, 255, 0.4)';
        sparkle.style.boxShadow = '0 0 8px 1px rgba(255, 255, 255, 0.3), inset 0 0 4px rgba(255, 255, 255, 0.2)';
        sparkle.style.border = '0.5px solid rgba(255, 255, 255, 0.2)';
        const size = Math.random() * 5 + 4;
        sparkle.style.width = `${size}px`;
        sparkle.style.height = `${size}px`;
        document.body.appendChild(sparkle);
        setTimeout(() => {
          if (sparkle.parentNode) sparkle.parentNode.removeChild(sparkle);
        }, 800);
      }
    };
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  const dockedSoFar = data && episodeIdx >= 0
    ? data.episodes.slice(0, episodeIdx + 1).filter(e => e.docked).length
    : 0;

  const currentEp = data && episodeIdx >= 0 ? data.episodes[episodeIdx] : null;

  // Effect 1: Simulation Step Timer (The 'Flight' Loop)
  useEffect(() => {
    if (!isPlaying || !data || !currentEp) return;
    if (stepIdx >= currentEp.trajectory.length - 1) return;

    const delay = Math.max(20, 500 / speed);
    const timer = setTimeout(() => {
      setStepIdx(prev => Math.min(currentEp.trajectory.length - 1, prev + 1));
    }, delay);

    return () => clearTimeout(timer);
  }, [isPlaying, episodeIdx, stepIdx, speed, data, currentEp]);

  // Effect 2: Episode Transition (The 'Sequence' Loop)
  useEffect(() => {
    if (!isPlaying || !data || !currentEp) return;
    
    // Trigger when we reach the absolute end of a trajectory
    if (stepIdx === currentEp.trajectory.length - 1) {
      // 1. Show final status
      setStatus({
        type: currentEp.docked ? 'success' : 'fail',
        text: `${currentEp.docked ? 'DOCKED' : 'FAILED'} — NEXT MISSION IN 1.5S...`
      });

      // 2. Schedule next episode
      const timer = setTimeout(() => {
        if (episodeIdx + 1 >= data.episodes.length) {
          setIsPlaying(false);
          setStatus({ type: 'success', text: 'ALL MISSIONS COMPLETE' });
          return;
        }
        
        // Move to next
        setEpisodeIdx(prev => prev + 1);
        setStepIdx(0);
        setStatus({ type: 'running', text: 'PLAYING' });
      }, 1500);

      return () => clearTimeout(timer);
    }
  }, [isPlaying, episodeIdx, stepIdx, data, currentEp]);

  const handleFileLoad = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        const parsed = JSON.parse(ev.target.result);
        // Safety Sort: ensure episodes are in increasing order by episode number
        if (parsed.episodes) {
          parsed.episodes.sort((a, b) => Number(a.episode) - Number(b.episode));
        }
        setData(parsed);
        setEpisodeIdx(0);
        setStepIdx(0);
        setIsPlaying(false);
        setStatus({ type: 'idle', text: 'LOADED' });
        if (timerRef.current) clearTimeout(timerRef.current);
        if (transitionTimerRef.current) clearTimeout(transitionTimerRef.current);
      } catch (err) {
        alert('Invalid JSON: ' + err.message);
      }
    };
    reader.readAsText(file);
  };

  const startPlayAll = () => {
    // 1. Force Stop everything
    setIsPlaying(false);
    if (timerRef.current) clearTimeout(timerRef.current);
    if (transitionTimerRef.current) clearTimeout(transitionTimerRef.current);
    setStatus({ type: 'idle', text: 'RESETTING...' });

    // 2. Clear state indices
    setEpisodeIdx(0);
    setStepIdx(0);

    // 3. Restart playback after indices are settled
    setTimeout(() => {
      setIsPlaying(true);
      setStatus({ type: 'running', text: 'PLAYING' });
    }, 100);
  };

  const jumpEpisode = (delta) => {
    if (!data) return;
    setIsPlaying(false);
    if (timerRef.current) clearTimeout(timerRef.current);
    if (transitionTimerRef.current) clearTimeout(transitionTimerRef.current);

    setEpisodeIdx(prev => {
      const next = prev + delta;
      return Math.max(0, Math.min(data.episodes.length - 1, next));
    });
    setStepIdx(0);
  };

  const togglePlay = () => {
    if (isPlaying) {
      setIsPlaying(false);
      setStatus({ type: 'idle', text: 'PAUSED' });
      if (timerRef.current) clearTimeout(timerRef.current);
      if (transitionTimerRef.current) clearTimeout(transitionTimerRef.current);
    } else {
      setIsPlaying(true);
      setStatus({ type: 'running', text: 'PLAYING' });
    }
  };

  return (
    <div className="main-wrapper">
      {/* 1st Slide: Hero Section */}
      <div className="hero-section">
        <video
          className="hero-video"
          autoPlay
          loop
          muted
          playsInline
          src="/Spacecraft_docking.mp4"
        ></video>

        <img src="/banner.png" alt="Hackathon Banner" className="hero-corner-banner" />
        <div className="hero-overlay">
          <h1 className="hero-title">Project Amaze</h1>
        </div>
        <div
          className="scroll-indicator"
          onClick={() => document.getElementById('dashboard').scrollIntoView({ behavior: 'smooth' })}
          style={{ cursor: 'pointer' }}
        >
          v
        </div>
      </div>

      {/* 2nd Slide: Dashboard */}
      <div id="dashboard" className="app-container" style={{ position: 'relative' }}>
        {/* Deep Space Background Layer */}
        <div className="stars"></div>


        {/* Use the provided static image for the solar system background */}
        <img className="dashboard-bg" src="/whatsapp_bg.jpeg" alt="Solar System Background" />

        {/* Foreground Dashboard Elements */}

        <StatsRow
          data={data}
          episodeIdx={episodeIdx}
          dockedSoFar={dockedSoFar}
          totalReward={currentEp ? currentEp.total_reward : 0}
          stepsTaken={currentEp ? currentEp.steps_taken : 0}
          fuelRemaining={currentEp ? currentEp.fuel_remaining_pct : 0}
        />

        <Controls
          data={data}
          onFileLoad={handleFileLoad}
          isPlaying={isPlaying}
          onPlayAll={startPlayAll}
          onPauseResume={togglePlay}
          onJumpEpisode={jumpEpisode}
          episodeIdx={episodeIdx}
          setSpeed={setSpeed}
          speed={speed}
          status={status}
        />

        <div className="main-grid">
          <OrbitalCanvas data={data} episodeIdx={episodeIdx} stepIdx={stepIdx} />
          <ReasoningPanel data={data} episodeIdx={episodeIdx} stepIdx={stepIdx} />
          <RewardChart data={data} currentEpisodeIdx={episodeIdx} />
        </div>
      </div>

    </div>
  );
}

export default App;
