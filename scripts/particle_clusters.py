import sys
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from db.db_util import get_data

df = get_data()

bins = df[[
    'bin_0_3_0_5',
    'bin_0_5_1_0',
    'bin_1_0_2_5',
    'bin_2_5_5_0',
    'bin_5_0_10_0'
]]

kmeans = KMeans(n_clusters=3, random_state=42)

df['cluster'] = kmeans.fit_predict(bins)

plt.scatter(df['pm2_5'], df['pm10_0'], c=df['cluster'])

plt.xlabel("PM2.5")
plt.ylabel("PM10")
plt.title("Pollution Clusters")

plt.tight_layout()
plt.savefig("plots/pollution_clusters.png")
plt.show()