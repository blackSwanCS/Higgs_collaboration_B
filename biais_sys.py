import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Charger les données
df = pd.read_parquet('blackSwan_data/blackSwan_data.parquet')
df = df.sample(n=10000, random_state=42)

# Définir les biais
sigma_tes = 0.01
sigma_jes = 0.01
alpha_tes = 1 + sigma_tes
alpha_jes = 1 + sigma_jes

# Fonction pour calculer le quadrivecteur
def four_momentum(pt, eta, phi):
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e = pt * np.cosh(eta)
    return np.array([px, py, pz, e])

# Appliquer les biais et recalculer les variables
def apply_biased_event(event):
    # Quadrivecteurs originaux
    P_lep = four_momentum(event['PRI_lep_pt'], event['PRI_lep_eta'], event['PRI_lep_phi'])
    P_had = four_momentum(event['PRI_had_pt'], event['PRI_had_eta'], event['PRI_had_phi'])
    P_jet_leading = four_momentum(event['PRI_jet_leading_pt'], event['PRI_jet_leading_eta'], event['PRI_jet_leading_phi'])
    P_jet_subleading = four_momentum(event['PRI_jet_subleading_pt'], event['PRI_jet_subleading_eta'], event['PRI_jet_subleading_phi'])

    # Appliquer les biais
    P_had_biased = alpha_tes * P_had
    P_jet_leading_biased = alpha_jes * P_jet_leading
    P_jet_subleading_biased = alpha_jes * P_jet_subleading

    # Recalculer les pt biaisés
    PRI_had_pt_biased = np.linalg.norm(P_had_biased[:2])
    PRI_jet_leading_pt_biased = np.linalg.norm(P_jet_leading_biased[:2])
    PRI_jet_subleading_pt_biased = np.linalg.norm(P_jet_subleading_biased[:2])
    PRI_jet_all_pt_biased = PRI_jet_leading_pt_biased + PRI_jet_subleading_pt_biased

    # Recalculer le MET biaisé
    P_visible_biased = P_lep + P_had_biased + P_jet_leading_biased + P_jet_subleading_biased
    P_MET_biased = -P_visible_biased
    PRI_met_biased = np.linalg.norm(P_MET_biased[:2])
    PRI_met_phi_biased = np.arctan2(P_MET_biased[1], P_MET_biased[0])

    # Recalculer DER_mass_vis biaisé
    P_vis_biased = P_lep + P_had_biased
    DER_mass_vis_biased = np.sqrt(np.maximum(0, P_vis_biased[3]**2 - np.sum(P_vis_biased[:3]**2)))

    # Recalculer DER_pt_h biaisé
    P_h_biased = P_had_biased + P_jet_leading_biased + P_jet_subleading_biased
    DER_pt_h_biased = np.linalg.norm(P_h_biased[:2])

    return {
        'PRI_had_pt_biased': PRI_had_pt_biased,
        'PRI_jet_leading_pt_biased': PRI_jet_leading_pt_biased,
        'PRI_jet_subleading_pt_biased': PRI_jet_subleading_pt_biased,
        'PRI_jet_all_pt_biased': PRI_jet_all_pt_biased,
        'PRI_met_biased': PRI_met_biased,
        'PRI_met_phi_biased': PRI_met_phi_biased,
        'DER_mass_vis_biased': DER_mass_vis_biased,
        'DER_pt_h_biased': DER_pt_h_biased
    }

# Appliquer à l'ensemble du DataFrame
biased_data = df.apply(apply_biased_event, axis=1, result_type='expand')
df_biased = pd.concat([df, biased_data], axis=1)

# Fonction pour tracer les histogrammes
def plot_feature_shift(df, feature):
    biased_feature = feature + '_biased'
    if biased_feature not in df.columns:
        print(f"⚠ La feature '{biased_feature}' n'est pas disponible.")
        return

    original = df[feature]
    biased = df[biased_feature]

    # Calcul du shift normalisé à la moyenne
    mean_original = original.mean()
    mean_biased = biased.mean()
    shift = (mean_biased - mean_original) / mean_original

    plt.figure(figsize=(10, 6))
    plt.hist(original, bins=60, density=True, alpha=0.5, label='Original')
    plt.hist(biased, bins=60, density=True, alpha=0.5, label='Biaisé')
    plt.axvline(mean_original, color='blue', linestyle='--', label=f'Moyenne originale: {mean_original:.2f}')
    plt.axvline(mean_biased, color='red', linestyle='--', label=f'Moyenne biaisée: {mean_biased:.2f}')
    plt.title(f"Distribution de '{feature}' avant et après biais")
    plt.xlabel(feature)
    plt.ylabel("Densité")
    plt.legend()
    plt.text(0.95, 0.95, f"Shift = {shift:.2%}", ha='right', va='top',
             transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_feature_shift(df_biased, 'DER_mass_vis')