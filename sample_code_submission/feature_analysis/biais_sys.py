import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from HiggsML.systematics import systematics

# Charger les données
df = pd.read_parquet("blackSwan_data/blackSwan_data.parquet")
pt_columns = ["PRI_lep_pt", "PRI_had_pt", "PRI_jet_leading_pt", "PRI_jet_subleading_pt"]
df = df[(df[pt_columns] > 0).all(axis=1)]
df = df.sample(n=10000, random_state=42)

# Sauvegarder version non biaisée
df_original = df.copy()

# Définir et appliquer les biais
sigma_tes = 0.01
sigma_jes = 0.01
alpha_tes = 1 + sigma_tes
alpha_jes = 1 + sigma_jes
df_biased = systematics(df.copy(), tes=alpha_tes, jes=alpha_jes)
df_biased = df_biased[(df[pt_columns] > 0).all(axis=1)]

# Liste complète des features
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


# Fonction de tracé
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


# Tracer un exemple
plot_feature_shift(df_original, df_biased, "DER_lep_eta_centrality")

# Dictionnaire des shifts
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

# Affichage final
print("\n=== SHIFT DICTIONARY (écart quadratique moyen relatif, en %) ===")
for f, s in shift_dict.items():
    if s is not np.nan:
        print(f"{f} : {s * 100:.4f}%")
