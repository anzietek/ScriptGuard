import sys
import os

os.environ['POSTGRES_PASSWORD'] = 'scriptguard_secure_password'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from scriptguard.database import DatasetManager

print("Connecting with password: scriptguard_secure_password")

db = DatasetManager()
samples = db.get_all_samples(limit=None)
malicious = [s for s in samples if s.get('label') == 'malicious']
benign = [s for s in samples if s.get('label') == 'benign']

print("=" * 70)
print("FRESH DATABASE STATUS")
print("=" * 70)
print(f"Total samples:     {len(samples)}")
print(f"Malicious:         {len(malicious)}")
print(f"Benign:            {len(benign)}")
print("=" * 70)

if len(samples) == 0:
    print("\n[SUCCESS] Database is EMPTY and FRESH!")
    print("Ready for ingestion testing!")
else:
    print(f"\n[WARNING] Database contains {len(samples)} samples")
