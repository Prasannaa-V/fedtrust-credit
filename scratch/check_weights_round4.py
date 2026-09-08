import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_partition import load_and_clean, partition_noniid, get_client_splits, build_shared_feature_columns
from local_training import FederatedClient
from consistency_score import per_client_agreement
from aggregation_strategy import explanation_consistency_aware_aggregate

df = load_and_clean(str(ROOT / "data" / "lending_club" / "loan.csv"))
partitions = partition_noniid(df)
shared_cols = build_shared_feature_columns(partitions)

clients = []
client_sizes = []
for cid in ["client_1", "client_2", "client_3"]:
    cdf = partitions[cid]
    X_tr, X_te, y_tr, y_te = get_client_splits(cdf, feature_cols=shared_cols)
    c = FederatedClient(client_id=cid, X_train=X_tr, X_test=X_te, y_train=y_tr, y_test=y_te)
    clients.append(c)
    client_sizes.append(len(y_tr))

global_params = [np.array([0.5], dtype=np.float32)]

for rnd in range(1, 5):
    print(f"Fitting round {rnd}...")
    params_list = []
    shap_list = []
    for c in clients:
        p, n, m = c.fit(global_params, config={"round": rnd})
        params_list.append(p[0])
        shap_list.append(np.array(m["shap_vector"], dtype=np.float32))

    if rnd == 4:
        agreements = per_client_agreement(shap_list)
        total_sz = sum(client_sizes)
        base_weights = [sz / total_sz for sz in client_sizes]
        print(f"\n==================== ROUND 4 WEIGHT SANITY CHECK ====================")
        print(f"Client Sizes: {client_sizes}")
        print(f"Base Weights (FedAvg, size-proportional):")
        for i, (cid, bw) in enumerate(zip(["client_1", "client_2", "client_3"], base_weights)):
            print(f"  {cid}: {bw:.6f} ({bw*100:.2f}%)")

        print(f"\nPer-Client Agreement terms (mean cosine similarity to other clients):")
        for i, (cid, ag) in enumerate(zip(["client_1", "client_2", "client_3"], agreements)):
            print(f"  {cid}: {ag:.6f}")

        print(f"\nWeight adjustment across gain values:")
        for g in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
            adj_unnorm = [base_weights[i] * (1.0 + g * agreements[i]) for i in range(3)]
            tot = sum(adj_unnorm)
            adj_norm = [w / tot for w in adj_unnorm]
            print(f"\n--- GAIN = {g:4.1f} ---")
            for i, cid in enumerate(["client_1", "client_2", "client_3"]):
                delta = adj_norm[i] - base_weights[i]
                pct_change = (delta / base_weights[i]) * 100
                print(f"  {cid}: base={base_weights[i]:.6f} -> adjusted={adj_norm[i]:.6f} | delta={delta:+.6f} ({pct_change:+.3f}%)")
        print("====================================================================\n")

    agg, _ = explanation_consistency_aware_aggregate(params_list, client_sizes, shap_list, consistency_gain=1.0)
    global_params = [agg]
