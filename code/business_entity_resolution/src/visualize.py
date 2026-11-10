import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 8.5
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.35
plt.rcParams['grid.linestyle'] = '--'

def plot_threshold_calibration(thresholds, precisions, recalls, singletons, f05_scores, best_tau=0.74, save_path=None):
    """Plot Fig 1: Decision Threshold Calibration Curve."""
    fig, ax1 = plt.subplots(figsize=(4.2, 3.1), dpi=300)
    tau = np.array(thresholds)
    
    ax1.plot(tau, precisions, 's--', color='#1B4F72', label='Precision (%)', linewidth=1.5, markersize=4)
    ax1.plot(tau, singletons, '^-.', color='#117A65', label='Singleton Acc. (%)', linewidth=1.5, markersize=4)
    ax1.plot(tau, recalls, 'd:', color='#78281F', label='Recall (%)', linewidth=1.5, markersize=4)
    ax1.plot(tau, f05_scores, 'o-', color='#B03A2E', label=r'Macro-$F_{0.5}$ (%)', linewidth=2.2, markersize=5)

    ax1.axvline(x=best_tau, color='#283747', linestyle=':', linewidth=1.5)
    best_idx = np.argmin(np.abs(tau - best_tau))
    ax1.annotate(f'$\\tau^* = {best_tau:.2f}$\n({f05_scores[best_idx]:.2f}%)', 
                 xy=(best_tau, f05_scores[best_idx]), xytext=(best_tau - 0.15, f05_scores[best_idx] - 4.5),
                 arrowprops=dict(facecolor='#283747', shrink=0.08, width=1, headwidth=4),
                 fontsize=8.5, fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', facecolor='#FADBD8', alpha=0.8))

    ax1.set_xlabel(r'Decision Probability Threshold ($\tau$)')
    ax1.set_ylabel('Metric Score (%)')
    ax1.set_ylim(80, 102)
    ax1.legend(loc='lower left', framealpha=0.9)
    plt.title(r'Fig. 1: Metric Sensitivity vs. Threshold $\tau$', fontsize=10.5, pad=6)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()

def plot_blocking_tradeoff(methods, recall_vals, latency_vals, save_path=None):
    """Plot Fig 2: Candidate Blocking Trade-Off (Recall vs. Latency)."""
    fig, ax1 = plt.subplots(figsize=(4.2, 3.1), dpi=300)
    x = np.arange(len(methods))
    width = 0.38
    color_rec, color_lat = '#2980B9', '#E67E22'

    ax1.bar(x - width/2, recall_vals, width, label='Recall (PC %)', color=color_rec, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Pairs Completeness (Recall %)', color=color_rec)
    ax1.tick_params(axis='y', labelcolor=color_rec)
    ax1.set_ylim(85, 100)

    ax2 = ax1.twinx()
    ax2.bar(x + width/2, latency_vals, width, label='Latency (s)', color=color_lat, edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Execution Latency (s) [log scale]', color=color_lat)
    ax2.tick_params(axis='y', labelcolor=color_lat)
    ax2.set_yscale('log')
    ax2.set_ylim(1, 200)

    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=8.5)
    plt.title('Fig. 2: Candidate Blocking Benchmark (N=15,000)', fontsize=10.5, pad=6)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()

def plot_feature_importance(feature_names, importance_vals, save_path=None):
    """Plot Fig 3: 11-Dimensional Feature Saliency Ranking."""
    fig, ax = plt.subplots(figsize=(4.2, 3.3), dpi=300)
    # Sort ascending
    paired = sorted(zip(feature_names, importance_vals), key=lambda x: x[1])
    feats = [p[0] for p in paired]
    vals = [p[1] for p in paired]
    colors = ['#1F618D' if v >= 10.0 else '#5DADE2' for v in vals]

    bars = ax.barh(feats, vals, color=colors, edgecolor='black', linewidth=0.5, height=0.65)
    ax.set_xlabel('Relative Feature Importance (Gini %)')
    ax.set_xlim(0, max(vals) * 1.2)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.5, bar.get_y() + bar.get_height()/2, f'{w:.1f}%', va='center', ha='left', fontsize=8)

    plt.title('Fig. 3: LightGBM Feature Saliency Ranking', fontsize=10.5, pad=6)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()

def plot_latency_breakdown(stages, durations, save_path=None):
    """Plot Fig 4: End-to-End Pipeline Execution Latency Share."""
    fig, ax = plt.subplots(figsize=(4.0, 3.1), dpi=300)
    colors = ['#2E86C1', '#5DADE2', '#48C9B0', '#F5B041', '#EB984E']

    wedges, texts, autotexts = ax.pie(
        durations, labels=stages, autopct='%1.1f%%', pctdistance=0.75,
        colors=colors, startangle=140,
        wedgeprops=dict(width=0.45, edgecolor='black', linewidth=0.6)
    )
    for t in texts:
        t.set_fontsize(7.5)
    for at in autotexts:
        at.set_fontsize(7.5)
        at.set_fontweight('bold')

    total_min = sum(durations) / 60.0
    ax.text(0, 0, f'Total\n{total_min:.1f} min\n(1.73M)', ha='center', va='center', fontsize=8.5, fontweight='bold', color='#1B4F72')
    plt.title('Fig. 4: Test Corpus Runtime Distribution', fontsize=10.5, pad=6)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()

def plot_model_comparison(models, precisions, f1_scores, f05_scores, save_path=None):
    """Plot Fig 5: Model Classification Benchmark Comparison."""
    fig, ax = plt.subplots(figsize=(4.4, 3.1), dpi=300)
    x = np.arange(len(models))
    w = 0.25

    ax.bar(x - w, precisions, w, label='Precision (%)', color='#5DADE2', edgecolor='black', linewidth=0.5)
    ax.bar(x, f1_scores, w, label=r'Macro-$F_1$ (%)', color='#48C9B0', edgecolor='black', linewidth=0.5)
    ax.bar(x + w, f05_scores, w, label=r'Macro-$F_{0.5}$ (%)', color='#C0392B', edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Validation Score (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right', fontsize=8.5)
    ax.set_ylim(80, 102)
    ax.legend(loc='lower right', framealpha=0.9, fontsize=8)
    plt.title(r'Fig. 5: End-to-End Linkage Performance', fontsize=10.5, pad=6)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()

def plot_jurisdiction_reduction(categories, comparisons_trillions, save_path=None):
    """Plot Fig 6: Sovereign Jurisdiction Comparison Space Reduction."""
    fig, ax = plt.subplots(figsize=(4.2, 3.1), dpi=300)
    bars = ax.bar(categories, comparisons_trillions, color=['#7B7D7D', '#27AE60'], edgecolor='black', linewidth=0.6, width=0.45)
    ax.set_ylabel('Pairwise Comparisons (Trillions)')
    ax.set_ylim(0, 20)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.4, f'{yval:.2f} T', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.annotate('-83.4% Reduction\n(0.0000% Error Invariant)',
                xy=(1, 3.2), xytext=(0.4, 12),
                arrowprops=dict(facecolor='#1E8449', shrink=0.08, width=1.5, headwidth=6),
                fontsize=8.5, fontweight='bold', color='#1E8449',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='#EAFAF1', edgecolor='#27AE60', alpha=0.9))

    plt.title('Fig. 6: Sovereign Jurisdiction Space Pruning', fontsize=10.5, pad=6)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()
