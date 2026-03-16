import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from db.db_util import get_data

df = get_data()

bins = [
    'bin_0_3_0_5',
    'bin_0_5_1_0',
    'bin_1_0_2_5',
    'bin_2_5_5_0',
    'bin_5_0_10_0'
]

df[bins].plot.area(figsize=(10,6))

plt.title("Particle Size Distribution")
plt.xlabel("Time Index")
plt.ylabel("Particle Count")

plt.tight_layout()
plt.savefig("plots/particle_distribution.png")
plt.show()