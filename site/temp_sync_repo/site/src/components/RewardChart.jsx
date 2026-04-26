import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';

ChartJS.register(
  CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler
);

function smoothRewards(arr, window) {
  return arr.map((_, i) => {
    const start = Math.max(0, i - window + 1);
    const slice = arr.slice(start, i + 1);
    return slice.reduce((a, b) => a + b, 0) / slice.length;
  });
}

export function RewardChart({ data, currentEpisodeIdx }) {
  if (!data || !data.reward_curve) {
    return (
      <div className="panel">
        <div className="panel-title">Reward Curve — Training Progress</div>
        <div id="reward-chart-container"></div>
        <div className="log-line">Load training_log.json to see reward curve</div>
      </div>
    );
  }

  const rewards = data.reward_curve;
  const labels = rewards.map((_, i) => i + 1);
  const smoothed = smoothRewards(rewards, 10);

  const chartData = {
    labels,
    datasets: [
      {
        label: 'Episode Reward',
        data: rewards,
        borderColor: 'rgba(59,130,246,0.3)',
        backgroundColor: 'transparent',
        borderWidth: 0.8,
        pointRadius: 0,
      },
      {
        label: 'Moving Average (10)',
        data: smoothed,
        borderColor: '#10b981',
        backgroundColor: 'rgba(16,185,129,0.1)',
        borderWidth: 2,
        pointRadius: 0,
        fill: true,
      },
    ]
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    plugins: {
      legend: {
        labels: { color: '#94a3b8', font: { size: 10 }, boxWidth: 12 }
      }
    },
    scales: {
      x: {
        title: { display: true, text: 'Training Episode', color: '#94a3b8', font: { size: 10 } },
        ticks: { color: '#64748b', maxTicksLimit: 10, font: { size: 9 } },
        grid: { color: 'rgba(42,58,85,0.5)' },
      },
      y: {
        title: { display: true, text: 'Episode Reward', color: '#94a3b8', font: { size: 10 } },
        ticks: { color: '#64748b', font: { size: 9 } },
        grid: { color: 'rgba(42,58,85,0.5)' },
      }
    }
  };

  return (
    <div className="panel">
      <div className="panel-title">Reward Curve — Training Progress</div>
      <div id="reward-chart-container">
        <Line data={chartData} options={options} />
      </div>
    </div>
  );
}
