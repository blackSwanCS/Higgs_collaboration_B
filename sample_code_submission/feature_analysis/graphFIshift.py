import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from HiggsML.systematics import systematics
import joblib

# -------- 1. Charger les données --------
df = pd.read_parquet('blackSwan_data/blackSwan_data.parquet')

# Remplacer les -25 (valeurs non physiques) par NaN
df.replace(-25, np.nan, inplace=True)

df = df.sample(n=10000, random_state=42)
df_original = df.copy()

sigma_tes = 0.01
sigma_jes = 0.01
alpha_tes = 1 + sigma_tes
alpha_jes = 1 + sigma_jes

# IMPORTANT : garder l’index d’origine
df.reset_index(drop=False, inplace=True)  # conserve l'ancien index dans une colonne 'index'
df_biased = systematics(df.copy(), tes=alpha_tes, jes=alpha_jes)

# Remettre l'index original sur df_biased pour aligner avec df_original
df_biased.set_index('index', inplace=True)

# Aligner df_original et df_biased sur le même index
common_idx = df_original.index.intersection(df_biased.index)
df_original_aligned = df_original.loc[common_idx]
df_biased_aligned = df_biased.loc[common_idx]



# Garder uniquement les index communs
common_index = df_original.index.intersection(df_biased.index)

# Aligner les deux DataFrames
df_original_aligned = df_original.loc[common_index].copy()
df_biased_aligned = df_biased.loc[common_index].copy()
print(df_original)
print(df_biased)

# -------- 4. Liste des features --------
features = [
    "PRI_lep_pt","PRI_lep_eta","PRI_lep_phi","PRI_had_pt","PRI_had_eta","PRI_had_phi",
    "PRI_jet_leading_pt","PRI_jet_leading_eta","PRI_jet_leading_phi","PRI_jet_subleading_pt",
    "PRI_jet_subleading_eta","PRI_jet_subleading_phi","PRI_n_jets","PRI_jet_all_pt","PRI_met",
    "PRI_met_phi","DER_mass_transverse_met_lep","DER_mass_vis","DER_pt_h","DER_deltaeta_jet_jet",
    "DER_mass_jet_jet","DER_prodeta_jet_jet","DER_deltar_had_lep","DER_pt_tot","DER_sum_pt",
    "DER_pt_ratio_lep_had","DER_met_phi_centrality","DER_lep_eta_centrality"
]

# -------- 5. Fonction de tracé --------
from scipy.stats import wasserstein_distance

def plot_feature_shift(df_orig, df_biased, feature):
    if feature not in df_orig.columns or feature not in df_biased.columns:
        print(f"⚠ Feature '{feature}' non trouvée.")
        return

    original = df_orig[feature]
    biased = df_biased[feature]
    mask = original.notna() & biased.notna()

    mean_orig = original[mask].mean()
    mean_biased = biased[mask].mean()

    # ➤ Remplacement : Calcul de la distance de Wasserstein
    shift_value = wasserstein_distance(original[mask], biased[mask])

    plt.figure(figsize=(10, 6))
    plt.hist(original[mask], bins=60, density=True, alpha=0.5, label='Original')
    plt.hist(biased[mask], bins=60, density=True, alpha=0.5, label='Biaisé')
    plt.axvline(mean_orig, color='blue', linestyle='--', label=f'Moy. Orig: {mean_orig:.2f}')
    plt.axvline(mean_biased, color='red', linestyle='--', label=f'Moy. Biaisé: {mean_biased:.2f}')
    plt.title(f"Distribution de '{feature}'")
    plt.xlabel(feature)
    plt.ylabel("Densité")
    plt.legend()
    plt.text(0.95, 0.95, f"Wasserstein = {shift_value:.4f}", ha='right', va='top',
             transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def clean_feature(orig, biased, threshold=10):
    """
    Nettoie les paires de valeurs originales/biaisées en supprimant les valeurs trop extrêmes.
    """
    mask = orig.notna() & biased.notna()
    orig_clean = orig[mask]
    biased_clean = biased[mask]

    # Supprimer les paires où l'une des deux valeurs est absurde
    valid_mask = (np.abs(orig_clean) < threshold) & (np.abs(biased_clean) < threshold)
    return orig_clean[valid_mask], biased_clean[valid_mask]



from scipy.stats import wasserstein_distance

# -------- 6. Calcul des shifts (Wasserstein) --------
shift_dict = {}
for feature in features:
    if feature in df_original.columns and feature in df_biased.columns:
        orig = df_original[feature]
        biased = df_biased[feature]
        orig_clean, biased_clean = clean_feature(orig, biased, threshold=20)
        if len(orig_clean) > 0 and len(biased_clean) > 0:
            shift = wasserstein_distance(orig_clean, biased_clean)
            shift_dict[feature] = shift
        else:
            shift_dict[feature] = np.nan
    else:
        shift_dict[feature] = np.nan
print(orig_clean)
print(biased_clean)
# -------- 7. Affichage des shifts --------
print("\n=== SHIFT DICTIONARY (Wasserstein distance) ===")
for f, s in shift_dict.items():
    if not pd.isna(s):
        print(f"{f} : {s:.4f}")

# -------- 8. Charger le modèle et importance --------
model = joblib.load("models/model.pkl")
feature_names = ['PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi', 'PRI_had_pt', 'PRI_had_eta',
       'PRI_had_phi', 'PRI_jet_leading_pt', 'PRI_jet_leading_eta',
       'PRI_jet_leading_phi', 'PRI_jet_subleading_pt',
       'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi', 'PRI_n_jets',
       'PRI_jet_all_pt', 'PRI_met', 'PRI_met_phi', 'weights',
       'detailed_labels', 'labels', 'DER_mass_transverse_met_lep',
       'DER_mass_vis', 'DER_pt_h', 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet',
       'DER_prodeta_jet_jet', 'DER_deltar_had_lep', 'DER_pt_tot',
       'DER_sum_pt']

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
})

# -------- 9. Fusionner avec shift --------
shift_df = pd.DataFrame([
    {'feature': k, 'shift': v} for k, v in shift_dict.items() if not pd.isna(v)
])
merged_df = pd.merge(importance_df, shift_df, on='feature', how='inner')
merged_df = merged_df[merged_df['feature'] != 'PRI_met_phi']  # Exclure phi (instable)

# -------- 10. Scatter shift vs importance --------
plt.figure(figsize=(10, 6))
plt.scatter(merged_df['shift'], merged_df['importance'], alpha=0.7)
plt.xlabel("Shift (Wasserstein)")
plt.ylabel("Importance (XGBoost)")
plt.title("Shift vs Importance des variables")
plt.grid(True)

for _, row in merged_df.iterrows():
    plt.annotate(row['feature'], (row['shift'], row['importance']), fontsize=8, alpha=0.6)

plt.tight_layout()
plt.show()


plot_feature_shift(df_original,df_biased,'DER_prodeta_jet_jet')