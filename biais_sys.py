import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Charger les données 
df = pd.read_parquet('blackSwan_data/blackSwan_data.parquet')

# Réduire à 10 000 événements
df = df.sample(n=10000, random_state=42)

# Définition des biais (1σ de l'article)
sigma_tes = 0.01  # TES: Tau Energy Scale
sigma_jes = 0.01  # JES: Jet Energy Scale
alpha_tes = 1 + sigma_tes
alpha_jes = 1 + sigma_jes

def four_momentum(pt, eta, phi):
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e = pt * np.cosh(eta)
    return np.array([px, py, pz, e])

def apply_biased_event(event, alpha_tes, alpha_jes):
    P_had = four_momentum(event['PRI_had_pt'], event['PRI_had_eta'], event['PRI_had_phi'])
    P_jet_leading = four_momentum(event['PRI_jet_leading_pt'], event['PRI_jet_leading_eta'], event['PRI_jet_leading_phi'])
    P_jet_subleading = four_momentum(event['PRI_jet_subleading_pt'], event['PRI_jet_subleading_eta'], event['PRI_jet_subleading_phi'])

    P_had_biased = alpha_tes * P_had
    P_jet_leading_biased = alpha_jes * P_jet_leading
    P_jet_subleading_biased = alpha_jes * P_jet_subleading

    return {
        'PRI_had_pt_biased': np.linalg.norm(P_had_biased[:2]),
        'PRI_jet_leading_pt_biased': np.linalg.norm(P_jet_leading_biased[:2]),
        'PRI_jet_subleading_pt_biased': np.linalg.norm(P_jet_subleading_biased[:2]),
    }

def apply_bias_to_df(df, alpha_tes, alpha_jes):
    biased_features = []
    for _, event in df.iterrows():
        biased_features.append(apply_biased_event(event, alpha_tes, alpha_jes))
    biased_df = pd.DataFrame(biased_features, index=df.index)
    return pd.concat([df, biased_df], axis=1)

# Appliquer le biais
df_biased = apply_bias_to_df(df, alpha_tes, alpha_jes)

# Nettoyage pour traçage
df_biased = df_biased.dropna(subset=['PRI_jet_leading_pt', 'PRI_jet_leading_pt_biased'])
df_biased['PRI_jet_leading_pt'] = df_biased['PRI_jet_leading_pt'].astype(float)
df_biased['PRI_jet_leading_pt_biased'] = df_biased['PRI_jet_leading_pt_biased'].astype(float)

# Moyennes
mean_orig = df_biased['PRI_jet_leading_pt'].mean()
mean_biased = df_biased['PRI_jet_leading_pt_biased'].mean()

# Tracé
plt.figure(figsize=(10, 6))
plt.hist(df_biased['PRI_jet_leading_pt'], bins=60, density=True, alpha=0.5, label='Original')
plt.hist(df_biased['PRI_jet_leading_pt_biased'], bins=60, density=True, alpha=0.5, label='Biased (JES +1σ)')
plt.axvline(mean_orig, color='blue', linestyle='--', label=f'Mean Original: {mean_orig:.2f}')
plt.axvline(mean_biased, color='red', linestyle='--', label=f'Mean Biased: {mean_biased:.2f}')
plt.title("Histogram of 'PRI_jet_leading_pt' with JES Bias (+1σ)")
plt.xlabel("Transverse Momentum (GeV)")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
