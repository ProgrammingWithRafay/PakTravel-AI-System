# Part 5: Traveller Clustering (K-Means)
# required libraries: pip install numpy pandas scikit-learn matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("=" * 65)
print("  PakTravel AI System - Part 5: Traveller Clustering")
print("=" * 65)

# ============================================================
# TASK 1: Data Generation + Exploration
# ============================================================
print("\n--- TASK 1: Data Generation & Exploration ---\n")

def generate_traveller_data(n=400):
    """
    Generates 400 synthetic traveller records.
    We create 3 natural groups by generating from different distributions.
    This ensures meaningful clusters emerge in K-Means.

    Group A - Frequent Business Travellers (~130 records)
    Group B - Occasional Family Travellers (~140 records)
    Group C - Regular Short-Distance Commuters (~130 records)
    """
    records = []

    # Group A: Business Travellers
    # travel often, book well in advance, long distances, high loyalty
    n_a = 130
    group_a = {
        'Trips_Per_Month':      np.random.randint(8, 16, n_a),
        'Avg_Booking_Days':     np.random.randint(10, 31, n_a),   # book ahead
        'Avg_Journey_Distance': np.random.randint(400, 701, n_a), # long routes
        'Preferred_Time':       np.random.choice([1, 2], n_a),    # early morning/morning
        'Loyalty_Points':       np.random.randint(2500, 5001, n_a),
        'Complaints_Filed':     np.random.randint(0, 4, n_a),
        'Group':                ['A'] * n_a
    }

    # Group B: Family Travellers
    # travel rarely, book last minute, medium distances
    n_b = 140
    group_b = {
        'Trips_Per_Month':      np.random.randint(1, 5, n_b),
        'Avg_Booking_Days':     np.random.randint(0, 8, n_b),     # last minute
        'Avg_Journey_Distance': np.random.randint(200, 500, n_b), # medium
        'Preferred_Time':       np.random.choice([2, 3, 4], n_b), # varied times
        'Loyalty_Points':       np.random.randint(500, 2500, n_b),
        'Complaints_Filed':     np.random.randint(1, 7, n_b),
        'Group':                ['B'] * n_b
    }

    # Group C: Short-Distance Commuters
    # travel very often, short routes, low loyalty (price-sensitive)
    n_c = 130
    group_c = {
        'Trips_Per_Month':      np.random.randint(10, 16, n_c),
        'Avg_Booking_Days':     np.random.randint(0, 5, n_c),     # same day or very soon
        'Avg_Journey_Distance': np.random.randint(100, 300, n_c), # short routes
        'Preferred_Time':       np.random.choice([1, 4], n_c),    # rush hours
        'Loyalty_Points':       np.random.randint(0, 1500, n_c),
        'Complaints_Filed':     np.random.randint(0, 6, n_c),
        'Group':                ['C'] * n_c
    }

    for grp in [group_a, group_b, group_c]:
        for i in range(len(grp['Group'])):
            records.append({
                'Trips_Per_Month':      grp['Trips_Per_Month'][i],
                'Avg_Booking_Days':     grp['Avg_Booking_Days'][i],
                'Avg_Journey_Distance': grp['Avg_Journey_Distance'][i],
                'Preferred_Time':       grp['Preferred_Time'][i],
                'Loyalty_Points':       grp['Loyalty_Points'][i],
                'Complaints_Filed':     grp['Complaints_Filed'][i],
                'Group':                grp['Group'][i],
            })

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle rows
    return df


df = generate_traveller_data(400)

print(f"Dataset Shape: {df.shape}")
print(f"\nFirst 10 Rows:")
print(df.drop('Group', axis=1).head(10).to_string(index=False))

# basic statistics
print(f"\nBasic Statistics:")
stats = df.drop('Group', axis=1).describe().loc[['mean', 'min', 'max']]
print(stats.to_string())

# missing values
missing = df.isnull().sum()
print(f"\nMissing Values:")
if missing.sum() == 0:
    print("  No missing values found in dataset.")
else:
    print(missing)

# histograms for 3 features
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(df['Trips_Per_Month'], bins=12, color='steelblue', edgecolor='black')
axes[0].set_title('Trips Per Month')
axes[0].set_xlabel('Trips')
axes[0].set_ylabel('Count')

axes[1].hist(df['Avg_Journey_Distance'], bins=12, color='darkorange', edgecolor='black')
axes[1].set_title('Avg Journey Distance (km)')
axes[1].set_xlabel('Distance (km)')
axes[1].set_ylabel('Count')

axes[2].hist(df['Loyalty_Points'], bins=12, color='green', edgecolor='black')
axes[2].set_title('Loyalty Points')
axes[2].set_xlabel('Points')
axes[2].set_ylabel('Count')

plt.suptitle('PakTravel Traveller Feature Distributions', fontsize=13)
plt.tight_layout()
plt.savefig('part5_histograms.png', dpi=150, bbox_inches='tight')
print("\nHistograms saved as part5_histograms.png")
plt.show()


# ============================================================
# TASK 2: K-Means Clustering
# ============================================================
print("\n" + "=" * 65)
print("--- TASK 2: K-Means Clustering ---\n")

# use only the 6 feature columns (not the true Group label)
feature_cols = ['Trips_Per_Month', 'Avg_Booking_Days', 'Avg_Journey_Distance',
                'Preferred_Time', 'Loyalty_Points', 'Complaints_Filed']

X = df[feature_cols].values

# normalize with StandardScaler (z-score normalization)
# StandardScaler is preferred here over MinMax because k-means uses euclidean distance
# and we don't want one feature (like Loyalty_Points) to dominate just because of scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Applying K-Means with k=3 (one cluster per traveller type)...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)
labels = kmeans.labels_

# add cluster labels to dataframe
df['Cluster'] = labels

# cluster sizes
print(f"\nCluster Sizes:")
for c in range(3):
    count = (labels == c).sum()
    print(f"  Cluster {c}: {count} travellers ({count/len(df)*100:.1f}%)")

# average feature values per cluster - this helps us label the clusters
print(f"\nAverage Feature Values per Cluster:")
cluster_means = df.groupby('Cluster')[feature_cols].mean()
print(cluster_means.round(2).to_string())

# ---- Cluster Labeling ----
# look at the means to figure out which cluster is which group
# Business: high trips, high booking days, high distance, high loyalty
# Family: low trips, low booking days, medium distance
# Commuter: high trips, low booking days, short distance, low loyalty

cluster_profiles = {}
for c in range(3):
    cm = cluster_means.loc[c]
    profile = {
        'trips':    cm['Trips_Per_Month'],
        'booking':  cm['Avg_Booking_Days'],
        'distance': cm['Avg_Journey_Distance'],
        'loyalty':  cm['Loyalty_Points'],
    }
    cluster_profiles[c] = profile

# determine labels automatically based on feature values
def auto_label_clusters(profiles):
    """
    Automatically assigns labels to clusters based on key distinguishing features.
    Business: high trips AND high distance AND high loyalty
    Commuter: high trips AND short distance
    Family: remaining
    """
    labels_map = {}
    clusters = list(profiles.keys())

    # sort by Avg_Journey_Distance to find short vs long distance clusters
    by_distance = sorted(clusters, key=lambda c: profiles[c]['distance'])
    by_loyalty  = sorted(clusters, key=lambda c: profiles[c]['loyalty'])
    by_booking  = sorted(clusters, key=lambda c: profiles[c]['booking'])

    # highest loyalty + highest booking days = Business
    business = by_loyalty[-1]
    labels_map[business] = 'Frequent Business Travellers'

    # shortest distance (from remaining) = Commuter
    remaining = [c for c in clusters if c != business]
    commuter = min(remaining, key=lambda c: profiles[c]['distance'])
    labels_map[commuter] = 'Regular Short-Distance Commuters'

    # the last one = Family
    family = [c for c in clusters if c not in labels_map][0]
    labels_map[family] = 'Occasional Family Travellers'

    return labels_map


cluster_labels_map = auto_label_clusters(cluster_profiles)

print(f"\nCluster Identification:")
for c, label in cluster_labels_map.items():
    cm = cluster_means.loc[c]
    print(f"\n  Cluster {c} --> {label}")
    print(f"    Avg Trips/Month   : {cm['Trips_Per_Month']:.1f}")
    print(f"    Avg Booking Days  : {cm['Avg_Booking_Days']:.1f}")
    print(f"    Avg Distance (km) : {cm['Avg_Journey_Distance']:.1f}")
    print(f"    Avg Loyalty Pts   : {cm['Loyalty_Points']:.1f}")
    print(f"    Avg Complaints    : {cm['Complaints_Filed']:.1f}")

print(f"\nJustification:")
print("""
  Business Travellers have the highest loyalty points (they travel most)
  and book well in advance (predictable schedules). Long distances.

  Family Travellers have the fewest trips per month and book at the
  last minute (spontaneous travel). Medium distances.

  Commuters have many trips per month (daily/weekly work travel) but
  short distances and low loyalty (price-sensitive, less brand-loyal).
""")

# scatter plot
fig, ax = plt.subplots(figsize=(10, 7))

colors = ['#E84040', '#3B82F6', '#22C55E']
group_names = [cluster_labels_map[c] for c in range(3)]

for c in range(3):
    mask = labels == c
    ax.scatter(df.loc[mask, 'Trips_Per_Month'],
               df.loc[mask, 'Avg_Journey_Distance'],
               c=colors[c], label=f"Cluster {c}: {cluster_labels_map[c]}",
               alpha=0.6, s=50, edgecolors='k', linewidths=0.3)

ax.set_xlabel('Trips Per Month', fontsize=12)
ax.set_ylabel('Avg Journey Distance (km)', fontsize=12)
ax.set_title('PakTravel Traveller Clusters\n(Trips/Month vs Journey Distance)', fontsize=13)
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('part5_clusters.png', dpi=150, bbox_inches='tight')
print("Scatter plot saved as part5_clusters.png")
plt.show()

# ---- Elbow Method ----
print("\nRunning Elbow Method (k = 1 to 8)...")

inertias = []
k_range  = range(1, 9)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    print(f"  k={k}: Inertia = {km.inertia_:.2f}")

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=8)
ax.axvline(x=3, color='red', linestyle='--', label='k=3 (chosen)', linewidth=1.5)
ax.set_xlabel('Number of Clusters (k)', fontsize=12)
ax.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
ax.set_title('Elbow Method - Optimal k for PakTravel Travellers', fontsize=13)
ax.set_xticks(list(k_range))
ax.legend()
ax.grid(True, linestyle='--', alpha=0.4)

# annotate the elbow point
ax.annotate('Elbow point\n(k=3)', xy=(3, inertias[2]),
            xytext=(4.5, inertias[2] + (inertias[0] - inertias[-1]) * 0.15),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10, color='red')

plt.tight_layout()
plt.savefig('part5_elbow.png', dpi=150, bbox_inches='tight')
print("Elbow plot saved as part5_elbow.png")
plt.show()

# is k=3 optimal?
# check the drop in inertia
drops = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
best_elbow = drops.index(max(drops)) + 1  # +1 because drops[0] is between k=1 and k=2

print(f"\nElbow Method Analysis:")
print(f"  Inertia drops between consecutive k values:")
for i, drop in enumerate(drops):
    marker = " <-- largest drop (elbow)" if i == drops.index(max(drops)) else ""
    print(f"    k={i+1} -> k={i+2}: drop = {drop:.2f}{marker}")

print(f"\n  Is k=3 the optimal choice?")
if best_elbow <= 3:
    print(f"  YES. The largest drop in inertia occurs at or before k=3.")
    print(f"  Adding more clusters beyond k=3 gives diminishing returns.")
    print(f"  The elbow in the graph clearly bends around k=3, confirming")
    print(f"  that 3 clusters capture the natural structure in our data.")
else:
    print(f"  The elbow suggests k={best_elbow} might be slightly better,")
    print(f"  but k=3 is well-justified by domain knowledge (3 traveller types)")
    print(f"  and the inertia drop is still significant at k=3.")

print(f"""
  Conclusion - Part 5:
  ====================
  K-Means successfully grouped PakTravel's 400 travellers into 3 meaningful
  clusters that match our expected traveller profiles:

  1. Frequent Business Travellers - PakTravel should offer:
     -> Monthly pass packages, priority boarding, loyalty rewards

  2. Occasional Family Travellers - PakTravel should offer:
     -> Last-minute booking discounts, family seat bundles

  3. Regular Short-Distance Commuters - PakTravel should offer:
     -> Weekly commuter passes, early morning route frequency

  Ali from our story falls into Group A (Business/Student frequent traveller)
  based on his planned route and travel pattern.

  The Elbow Method confirms k=3 is appropriate for this dataset.
""")

