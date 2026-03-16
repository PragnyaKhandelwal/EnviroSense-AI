import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from db.db_util import get_data

df = get_data()

plt.scatter(df['pm2_5_pcs'], df['pm2_5'])

plt.xlabel("Particle Count (PM2.5)")
plt.ylabel("Particle Mass (PM2.5)")
plt.title("Count vs Mass Relationship")

plt.tight_layout()
plt.savefig("plots/count_vs_mass.png")
plt.show()