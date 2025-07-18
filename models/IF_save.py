"""
IF_save.py — Train and save Isolation Forest model using final offline features,
then validate feature consistency against your existing RF model.
"""

import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest

# 1) Define the 33 features in the exact order:
SCALED_NUMERICAL_FEATURES = [
    'src_port',
    'dst_port',
    'duration',
    'max_packets_len',
    'skewness_packets_len',
    'variance_receiving_packets_len',
    'skewness_receiving_packets_len',
    'skewness_sending_packets_len',
    'min_receiving_packets_delta_len',
    'max_receiving_packets_delta_len',
    'mean_receiving_packets_delta_len',
    'median_receiving_packets_delta_len',
    'variance_receiving_packets_delta_len',
    'mode_receiving_packets_delta_len',
    'coefficient_of_variation_receiving_packets_delta_len',
    'skewness_receiving_packets_delta_len',
    'mean_sending_packets_delta_len',
    'median_sending_packets_delta_len',
    'mode_sending_packets_delta_len',
    'coefficient_of_variation_sending_packets_delta_len',
    'skewness_sending_packets_delta_len',
    'coefficient_of_variation_receiving_packets_delta_time',
    'skewness_sreceiving_packets_delta_time',
    'coefficient_of_variation_sending_packets_delta_time',
    'skewness_sending_packets_delta_time'
]
DNS_FEATURES      = ['dst_ip_freq', 'dns_domain_name_freq']
TEMPORAL_FEATURES = ['is_weekend', 'day_sin', 'day_cos']
SYNERGY_FEATURES  = ['cluster_size', 'distance_from_centroid', 'cluster_density']

FINAL_FEATURES = (
    SCALED_NUMERICAL_FEATURES +
    TEMPORAL_FEATURES +
    DNS_FEATURES +
    SYNERGY_FEATURES
)

# 2) Load offline CSV (it has extra columns we ignore)
df = pd.read_csv(
    "/Users/omar/Desktop/application/First year/Scripts/BCCC-CIC-Bell-DNS-2024/Test/offline_training_data_with_clusters.csv"
)

# 3) Extract exactly our 33 features, in that order
X = df[FINAL_FEATURES].values

# 4) Print the feature order and data shape
print("🚀 Training IsolationForest on these 33 features (in order):")
for i, feat in enumerate(FINAL_FEATURES, start=1):
    print(f"   {i:2d}. {feat}")
print(f"🔢 Training data shape: {X.shape} (n_samples, n_features)")

# 5) Train Isolation Forest
if_model = IsolationForest(
    n_estimators=420,
    max_samples=len(X),
    random_state=42,
    n_jobs=-1
)
if_model.fit(X)

# 6) Confirm model internals
print(f"✅ IF trained: n_estimators={if_model.n_estimators}, n_features_in_={if_model.n_features_in_}")

# 7) Save IF model
joblib.dump(if_model, "isoforest.joblib")
print("💾 Saved Isolation Forest to isoforest.joblib")

# 8) Load RF and validate its feature order
rf_model = joblib.load("best_offline_rf_model.joblib")
print(f"\n🔎 RF model n_features_in_ = {rf_model.n_features_in_}")

if hasattr(rf_model, 'feature_names_in_'):
    rf_feats = list(rf_model.feature_names_in_)
    print("RF model’s feature_names_in_:")
    for i, f in enumerate(rf_feats, start=1):
        print(f"   {i:2d}. {f}")
    if rf_feats == FINAL_FEATURES:
        print("✅ RF feature order matches FINAL_FEATURES exactly.")
    else:
        print("⚠️ RF feature order DOES NOT match FINAL_FEATURES!")
else:
    print("⚠️ RF model has no feature_names_in_; only feature counts will be compared.")

# 9) Compare feature counts
if (if_model.n_features_in_ == rf_model.n_features_in_ 
    == len(FINAL_FEATURES)):
    print(f"✅ Both IF and RF models use {len(FINAL_FEATURES)} features.")
else:
    print("⚠️ Feature count mismatch between IF and RF models.")

# 10) Final confirmation
print("\n✅ Validation and verification completed.")
