import sys
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_rewards(log_path: str, out_path: str | None = None) -> Path:
    '''
    plot the average reward of each search iteration from a .jsonl log.
    saves to out_path, by default next to the log with the same name and a .png extension.
    '''
    out_path = Path(out_path) if out_path else Path(log_path).with_suffix('.png')
    with open(log_path, encoding='utf-8') as f:
        logs = [json.loads(line) for line in f if line.strip()]

    iters = np.array([log['iter'] for log in logs])
    # iterations where every change failed have no scores -> nan (gap in the line)
    avg = np.array([np.mean(log['scores']) if log['scores'] else np.nan for log in logs])
    alpha = np.array([log['is_alpha'] for log in logs])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(iters, avg, color='tab:blue', marker='o', markersize=4, label='avg reward')
    ax.scatter(iters[alpha], avg[alpha], color='tab:orange', zorder=3, label='alpha model')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('iteration')
    ax.set_ylabel('average reward')
    ax.set_title(log_path.rsplit('/', 1)[-1])
    ax.grid(alpha=0.3)
    ax.legend()

    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


if __name__ == '__main__':
    print(plot_rewards(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))