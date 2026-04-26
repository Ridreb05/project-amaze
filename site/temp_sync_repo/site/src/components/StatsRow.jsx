export function StatsRow({ data, episodeIdx, dockedSoFar, totalReward, stepsTaken, fuelRemaining }) {
  const dockRate = episodeIdx >= 0 ? ((dockedSoFar / (episodeIdx + 1)) * 100).toFixed(0) : 0;
  
  return (
    <div className="stats-row">
      <div className="stat">
        <div className="stat-val">{data ? data.episodes[episodeIdx]?.episode : 0}</div>
        <div className="stat-lbl">Episode</div>
      </div>
      <div className="stat">
        <div className={`stat-val ${totalReward > 2 ? 'green' : totalReward < 0 ? 'red' : 'amber'}`}>
          {data ? totalReward.toFixed(2) : '—'}
        </div>
        <div className="stat-lbl">Episode Reward</div>
      </div>
      <div className="stat">
        <div className={`stat-val ${parseInt(dockRate) > 40 ? 'green' : parseInt(dockRate) > 15 ? 'amber' : 'red'}`}>
          {data ? dockRate + '%' : '0%'}
        </div>
        <div className="stat-lbl">Dock Rate</div>
      </div>
      <div className="stat">
        <div className="stat-val amber">{data ? fuelRemaining.toFixed(0) + '%' : '—'}</div>
        <div className="stat-lbl">Fuel Remaining</div>
      </div>
      <div className="stat">
        <div className="stat-val">{data ? stepsTaken : '—'}</div>
        <div className="stat-lbl">Steps Taken</div>
      </div>
    </div>
  );
}
