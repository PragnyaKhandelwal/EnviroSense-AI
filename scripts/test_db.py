import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from db.db_util import get_data

df = get_data()

print(df.head())
print("Rows:", len(df))
print(df.columns)