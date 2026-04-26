export function ReasoningPanel({ data, episodeIdx, stepIdx }) {
  if (!data || !data.episodes[episodeIdx]) {
    return (
      <div className="panel reasoning-panel">
        <div className="panel-title">Agent Reasoning — Step by Step</div>
        <div className="reasoning-box">
Waiting for episode data...

The agent will display its flight controller reasoning here as each step plays.
        </div>
      </div>
    );
  }

  const ep = data.episodes[episodeIdx];
  const step = ep.trajectory[stepIdx];
  
  if (!step) return null;

  const quality = step.reward > 0.5 ? 'good' : step.reward < -0.3 ? 'bad' : 'ok';
  const commsStatus = step.comms_active ? '' : '\n[COMMS BLACKOUT — flying blind]';
  const losStatus = step.los_violation ? '\n[LOS CONE VIOLATION]' : '';

  const reasoningText = `Episode ${ep.episode} | Step ${stepIdx + 1}/${ep.trajectory.length} | ${ep.difficulty}\n` +
    `─────────────────────────────────────────────────────\n` +
    `Thrust: fx=${step.fx.toFixed(3)}N, fy=${step.fy.toFixed(3)}N\n` +
    `Distance: ${step.distance_m.toFixed(2)}m | Fuel: ${step.fuel_pct.toFixed(0)}%\n` +
    `Step reward: ${step.reward >= 0 ? '+' : ''}${step.reward.toFixed(4)}${commsStatus}${losStatus}\n\n` +
    `Agent reasoning:\n"${step.reasoning || '[No reasoning provided]'}"`;

  return (
    <div className="panel reasoning-panel">
      <div className="panel-title">Agent Reasoning — Step by Step</div>
      <div className={`reasoning-box ${quality === 'bad' ? 'bad' : ''}`}>
        {reasoningText}
      </div>
    </div>
  );
}
