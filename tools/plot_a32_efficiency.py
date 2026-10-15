#!/usr/bin/env python3
"""Plot observed A32 quality/resource tradeoffs from completed aggregate evidence."""
import json
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['svg.hashsalt'] = 'reflex-a32-efficiency-v1'

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / 'releases/reflex-1'
NAMES = {
    'run-01': 'Reflex-1',
    'qwen-legacy-cached-fresh180': 'Base Qwen, cached app router',
}


def frontier(points, cost):
    return [p['id'] for p in points if not any(
        q[cost] <= p[cost] and q['accuracy'] >= p['accuracy']
        and (q[cost] < p[cost] or q['accuracy'] > p['accuracy'])
        for q in points if q is not p)]


def main():
    evidence = json.loads((RELEASE / 'benchmark-summary.json').read_text())
    points = []
    for record in evidence['completed_jobs']:
        key = record['directory']
        if key not in NAMES:
            continue
        summary = record['summary']
        total = summary['all_requests'] if key == 'run-01' else summary
        count = total.get('rows', total.get('total'))
        if count != 180 or total['failures']:
            raise ValueError('Expected completed, error-free fresh180 cohort')
        points.append(dict(id=key, label=NAMES[key], accuracy=100*total['accuracy'],
                           latency_s=total['mean_latency_ms']/1000,
                           sampled_pss_mib=record['resources']['sampled_max_pss_kib']/1024))
    if len(points) != 2:
        raise ValueError('Both fixed profiles are required')
    fronts = {cost: frontier(points, cost) for cost in ['latency_s', 'sampled_pss_mib']}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, cost, title, xlabel in zip(axes, fronts, ['Accuracy vs latency', 'Accuracy vs sampled RAM'],
                                      ['Mean request latency (seconds; lower is better)',
                                       'Sampled maximum process PSS (MiB; lower is better)']):
        selected = sorted([p for p in points if p['id'] in fronts[cost]], key=lambda p:p[cost])
        ax.plot([p[cost] for p in selected], [p['accuracy'] for p in selected],
                '--', color='#9aa6b2', linewidth=1, zorder=1, label='Observed frontier')
        for p in points:
            color = '#157f9e' if p['id']=='run-01' else '#d98c32' if p['id']=='qwen-fresh180' else '#596579'
            ax.scatter(p[cost], p['accuracy'], s=85, color=color, zorder=3)
            offset = (5, 10) if p['id']=='qwen-legacy-fresh180' else (5, -24)
            align_right = p['id']=='run-01' or cost=='latency_s'
            if p['id']=='run-01': offset=(-5, -32)
            elif align_right: offset=(-5,10)
            ax.annotate(p['label']+'\n'+f"{p['accuracy']:.1f}%", (p[cost],p['accuracy']),
                        xytext=offset, textcoords='offset points', fontsize=8,
                        ha='right' if align_right else 'left')
        ax.set(title=title, xlabel=xlabel, ylabel='Routing accuracy (%)', ylim=(0,100))
        lo, hi = min(p[cost] for p in points), max(p[cost] for p in points)
        ax.set_xlim(lo-(hi-lo)*.15, hi+(hi-lo)*.35)
        ax.grid(alpha=.2)
        ax.legend(loc='lower right', fontsize=8)
    fig.suptitle('MOBILE_ROUTING9 on Samsung Galaxy A32: base and Reflex-1', fontsize=13)
    fig.supxlabel('Same frozen 180 cases; CPU 4 threads, Q4_K_M. Single sustained run per profile.\n'
                  'Frontiers describe these observations only; dashed lines do not imply achievable intermediate models.', fontsize=9)
    assets = RELEASE/'assets'
    assets.mkdir(exist_ok=True)
    for suffix in ['png','svg']:
        target = assets/('a32-quality-efficiency.'+suffix)
        fig.savefig(target, dpi=180, **({'metadata':{'Date':None}} if suffix=='svg' else {}))
        if suffix == 'svg':
            target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines())+'\n')
    plt.close(fig)
    (RELEASE/'a32-efficiency.json').write_text(json.dumps({
        'points':points, 'observed_frontiers':fronts,
        'input_sha256':hashlib.sha256((RELEASE/'benchmark-summary.json').read_bytes()).hexdigest(),
        'generator':'tools/plot_a32_efficiency.py',
        'matplotlib_version':matplotlib.__version__,
        'definition':'Maximize accuracy and minimize the named cost; strict improvement in at least one objective.',
        'limitations':'Single runs; profile/policy and starting temperature differ. Sampled PSS is not in-flight peak memory. No measured energy or cross-model Decision Index frontier.'
    },indent=2)+'\n')


if __name__ == '__main__':
    main()
