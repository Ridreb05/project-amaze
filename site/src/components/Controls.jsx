export function Controls({
  data, onFileLoad, isPlaying, onPlayAll, onPauseResume,
  onJumpEpisode, episodeIdx, setSpeed, speed, status
}) {
  return (
    <div className="controls">
      <label className="btn" style={{ cursor: 'pointer' }}>
        <input type="file" className="hidden" accept=".json" onChange={onFileLoad} />
        Load Log
      </label>
      <button className="btn" onClick={onPlayAll} disabled={!data}>Play All</button>
      <button className="btn" onClick={onPauseResume} disabled={!data}>
        {isPlaying ? 'Pause' : 'Resume'}
      </button>
      <button className="btn" onClick={() => onJumpEpisode(-1)} disabled={!data || episodeIdx === 0}>Prev</button>
      <button className="btn" onClick={() => onJumpEpisode(1)} disabled={!data || episodeIdx === data.episodes.length - 1}>Next</button>
      <div className="speed-control">
        <span>Speed</span>
        <input type="range" min="1" max="20" value={speed} onChange={e => setSpeed(Number(e.target.value))} />
        <span className="speed-val">{speed}x</span>
      </div>
      <select
        value={episodeIdx}
        onChange={(e) => onJumpEpisode(Number(e.target.value) - episodeIdx)}
        disabled={!data}
      >
        <option value="-1">Jump to episode...</option>
        {data && data.episodes.map((ep, i) => (
          <option key={i} value={i}>Ep {ep.episode} — {ep.difficulty} — {ep.docked ? 'DOCKED' : 'failed'}</option>
        ))}
      </select>
      <span className={`status-badge badge-${status.type}`}>
        {status.text}
      </span>
      <span id="episode-label">{data ? `${data.episodes.length} episodes loaded` : 'No data loaded'}</span>
    </div>
  );
}
