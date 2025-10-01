#report.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import gaussian_kde
from matplotlib.ticker import AutoMinorLocator
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from matplotlib.colors import ListedColormap

from config import OUTPUT_CSV, REPORT_PDF

def make_report(csv_path=OUTPUT_CSV, pdf_path=REPORT_PDF):
    # load data
    df = pd.read_csv(csv_path)
    freq = df['peak_freq_khz'].dropna()
    dur  = df['duration_ms'].dropna()
    times = df['start_time_s']
    features = df[['peak_freq_khz','duration_ms']].dropna()
    X = StandardScaler().fit_transform(features)

    # clustering
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    labels = kmeans.labels_
    base_colors = plt.cm.tab10(np.arange(n_clusters))
    cmap = ListedColormap(base_colors)

    # create 3×2 layout (vertical orientation)
    fig, axes = plt.subplots(3, 2, figsize=(12, 18))

    # 1. Peak Frequency Density
    ax = axes[0, 0]
    kde = gaussian_kde(freq)
    x_vals = np.linspace(freq.min(), freq.max(), 200)
    ax.plot(x_vals, kde(x_vals), linewidth=2)
    ax.fill_between(x_vals, kde(x_vals), alpha=0.3)
    ax.set_title('Peak Frequency Density')
    ax.set_xlabel('Frequency (kHz)')
    ax.set_ylabel('Density')

    # 2. USV Duration Distribution
    ax = axes[0, 1]
    color_hist = '#1f77b4'
    color_kde  = '#ff7f0e'
    ax.hist(dur, bins=30, density=True, alpha=0.6,
            color=color_hist, edgecolor='white', label='Histogram')
    kde2 = gaussian_kde(dur)
    x2 = np.linspace(dur.min(), dur.max(), 200)
    ax.plot(x2, kde2(x2), color=color_kde, linewidth=2, label='KDE')
    med = np.median(dur)
    ax.axvline(med, color=color_kde, linestyle='--', linewidth=1, label='Median')
    ax.legend(frameon=False, fontsize=9)
    ax.set_title('USV Duration Distribution')
    ax.set_xlabel('Duration (ms)')
    ax.set_ylabel('Density')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis='x', which='minor', length=3)

    # 3. Event Density Over Time & Frequency
    ax = axes[1, 0]
    time_bins = np.linspace(times.min(), times.max(), 50)
    freq_bins = [0,30,40,50,60,80,100]
    h = ax.hist2d(times, freq, bins=[time_bins,freq_bins], cmap='plasma')
    ax.set_title('Event Density Over Time & Frequency')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Peak Frequency (kHz)')
    centers = (np.array(freq_bins[:-1]) + np.array(freq_bins[1:]))/2
    ax.set_yticks(centers)
    ax.set_yticklabels([f'{freq_bins[i]}–{freq_bins[i+1]}' 
                        for i in range(len(freq_bins)-1)])
    cb = plt.colorbar(h[3], ax=ax, label='Count')
    cb.ax.tick_params(labelsize=8)

    # 4. CDF of Peak Frequency
    ax = axes[1, 1]
    sorted_freq = np.sort(freq)
    cdf = np.arange(1, len(sorted_freq)+1) / len(sorted_freq)
    ax.plot(sorted_freq, cdf, linewidth=2)
    ax.set_title('CDF of Peak Frequency')
    ax.set_xlabel('Frequency (kHz)')
    ax.set_ylabel('Cumulative Probability')
    ax.grid(True, linestyle='--', alpha=0.4)

    # 5. Scatter plot: Clusters
    ax = axes[2, 0]
    sc = ax.scatter(features['peak_freq_khz'],
                    features['duration_ms'],
                    c=labels, cmap=cmap,
                    s=50, alpha=0.8,
                    edgecolor='k', linewidth=0.3)
    ax.set_title('USV Clusters: Frequency vs Duration')
    ax.set_xlabel('Peak Frequency (kHz)')
    ax.set_ylabel('Duration (ms)')
    ax.grid(True, linestyle='--', alpha=0.3)
    handles = [plt.Line2D([], [], marker='o', color=base_colors[i],
                          linestyle='None', markersize=6,
                          label=f'Cluster {i}')
               for i in range(n_clusters)]
    ax.legend(handles=handles, frameon=False, title='Cluster')

    # 6. Silhouette Plot
    ax = axes[2, 1]
    sil_vals = silhouette_samples(X, labels)
    y_lower = 10
    y_ticks = []
    for i, color in enumerate(base_colors):
        sv = np.sort(sil_vals[labels==i])
        size = sv.shape[0]
        y_upper = y_lower + size
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, sv, facecolor=color, alpha=0.6)
        y_ticks.append(y_lower + size/2)
        y_lower = y_upper + 10
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(i) for i in range(n_clusters)])
    avg = silhouette_score(X, labels)
    ax.axvline(avg, color='red', linestyle='--', label=f'Average = {avg:.2f}')
    ax.set_title('Silhouette Plot for KMeans')
    ax.set_xlabel('Silhouette Coefficient')
    ax.set_ylabel('Cluster')
    ax.legend(frameon=False, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_ylim(0, y_lower)

    # finalize and save
    fig.tight_layout(pad=2)
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig)
    plt.close(fig)

if __name__ == '__main__':
    make_report()
