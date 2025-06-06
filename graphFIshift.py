import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from HiggsML.systematics import systematics
import joblib

# -------- 1. Charger les données --------
df = pd.read_parquet("blackSwan_data/blackSwan_data.parquet")
pt_columns = ["PRI_lep_pt", "PRI_had_pt", "PRI_jet_leading_pt", "PRI_jet_subleading_pt"]
df = df[(df[pt_columns] > 0).all(axis=1)]
df = df.sample(n=10000, random_state=42)

# -------- 2. Sauvegarder version non biaisée --------
df_original = df.copy()

# -------- 3. Définir et appliquer les biais --------
sigma_tes = 0.01
sigma_jes = 0.01
alpha_tes = 1 + sigma_tes
alpha_jes = 1 + sigma_jes
df_biased = systematics(df.copy(), tes=alpha_tes, jes=alpha_jes)

# -------- 4. Liste des features --------
features = [
    "PRI_lep_pt",
    "PRI_lep_eta",
    "PRI_lep_phi",
    "PRI_had_pt",
    "PRI_had_eta",
    "PRI_had_phi",
    "PRI_jet_leading_pt",
    "PRI_jet_leading_eta",
    "PRI_jet_leading_phi",
    "PRI_jet_subleading_pt",
    "PRI_jet_subleading_eta",
    "PRI_jet_subleading_phi",
    "PRI_n_jets",
    "PRI_jet_all_pt",
    "PRI_met",
    "PRI_met_phi",
    "DER_mass_transverse_met_lep",
    "DER_mass_vis",
    "DER_pt_h",
    "DER_deltaeta_jet_jet",
    "DER_mass_jet_jet",
    "DER_prodeta_jet_jet",
    "DER_deltar_had_lep",
    "DER_pt_tot",
    "DER_sum_pt",
    "DER_pt_ratio_lep_had",
    "DER_met_phi_centrality",
    "DER_lep_eta_centrality",
]


# -------- 5. Fonction de tracé pour une seule feature --------
def plot_feature_shift(df_orig, df_biased, feature):
    if feature not in df_orig.columns or feature not in df_biased.columns:
        print(f"⚠ Feature '{feature}' non trouvée.")
        return

    original = df_orig[feature]
    biased = df_biased[feature]

    mask = (original > 0) & (biased > 0)
    mean_orig = original[mask].mean()
    mean_biased = biased[mask].mean()
    shift = ((biased[mask] - original[mask]) / (original[mask] + 1e-8)) ** 2
    shift_value = shift.mean()

    plt.figure(figsize=(10, 6))
    plt.hist(original, bins=60, density=True, alpha=0.5, label="Original")
    plt.hist(biased, bins=60, density=True, alpha=0.5, label="Biaisé")
    plt.axvline(
        mean_orig, color="blue", linestyle="--", label=f"Moy. Orig: {mean_orig:.2f}"
    )
    plt.axvline(
        mean_biased,
        color="red",
        linestyle="--",
        label=f"Moy. Biaisé: {mean_biased:.2f}",
    )
    plt.title(f"Distribution de '{feature}'")
    plt.xlabel(feature)
    plt.ylabel("Densité")
    plt.legend()
    plt.text(
        0.95,
        0.95,
        f"Shift = {shift_value*100:.2f}%",
        ha="right",
        va="top",
        transform=plt.gca().transAxes,
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.7),
    )
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Exemple : tracer la distribution biaisée vs originale
plot_feature_shift(df_original, df_biased, "DER_lep_eta_centrality")

# -------- 6. Calcul des shifts --------
shift_dict = {}
for feature in features:
    if feature in df_original.columns and feature in df_biased.columns:
        orig = df_original[feature]
        biased = df_biased[feature]
        mask = (orig > 0) & (biased > 0)
        if mask.sum() > 0:
            rel_squared_diff = ((biased[mask] - orig[mask]) / (orig[mask] + 1e-8)) ** 2
            shift_dict[feature] = rel_squared_diff.mean()
        else:
            shift_dict[feature] = np.nan
    else:
        shift_dict[feature] = np.nan

"""# Affichage des shifts
print("\n=== SHIFT DICTIONARY (écart quadratique moyen relatif, en %) ===")
for f, s in shift_dict.items():
    if s is not np.nan:
        print(f"{f} : {s * 100:.4f}%")"""

# -------- 7. Charger le modèle et importance --------
model = joblib.load("models/model.pkl")
feature_names = df.columns[: len(model.feature_importances_)]

importance_df = pd.DataFrame(
    {"feature": feature_names, "importance": model.feature_importances_}
)

# -------- 8. Fusionner avec shift --------
shift_df = pd.DataFrame(
    [{"feature": k, "shift": v} for k, v in shift_dict.items() if v is not np.nan]
)
merged_df = pd.merge(importance_df, shift_df, on="feature", how="inner")
merged_df = merged_df[merged_df["feature"] != "PRI_met_phi"]  # Exclure phi

# -------- 9. Affichage final : Scatter shift vs importance --------
plt.figure(figsize=(10, 6))
plt.scatter(merged_df["shift"], merged_df["importance"], alpha=0.7)
plt.xlabel("Shift (écart quadratique moyen relatif)")
plt.ylabel("Importance (XGBoost)")
plt.title("Shift vs Importance des variables")
plt.grid(True)

for _, row in merged_df.iterrows():
    plt.annotate(
        row["feature"], (row["shift"], row["importance"]), fontsize=8, alpha=0.6
    )

plt.tight_layout()
plt.show()
