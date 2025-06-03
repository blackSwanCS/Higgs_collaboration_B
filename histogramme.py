import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt

# Chemin vers ton fichier parquet (à adapter)
file_path = 'blackSwan_data/blackSwan_data.parquet'

# Chargement du dataset
df = pd.read_parquet(file_path)

# Liste des features à tracer
features = [
    "PRI_lep_pt","PRI_lep_eta","PRI_lep_phi","PRI_had_pt","PRI_had_eta","PRI_had_phi",
    "PRI_jet_leading_pt","PRI_jet_leading_eta","PRI_jet_leading_phi","PRI_jet_subleading_pt",
    "PRI_jet_subleading_eta","PRI_jet_subleading_phi","PRI_n_jets","PRI_jet_all_pt","PRI_met",
    "PRI_met_phi","DER_mass_transverse_met_lep","DER_mass_vis","DER_pt_h","DER_deltaeta_jet_jet",
    "DER_mass_jet_jet","DER_prodeta_jet_jet","DER_deltar_had_lep","DER_pt_tot","DER_sum_pt",
    "DER_pt_ratio_lep_had","DER_met_phi_centrality","DER_lep_eta_centrality"
]

# Séparer signal et background
df_signal = df[df['labels'] == 1]
df_background = df[df['labels'] == 0]



class MiniVisualiseur:
    def __init__(self, df_signal, df_background, features):
        self.df_signal = df_signal
        self.df_background = df_background
        self.features = features
        self.index = 0
        
        self.fig, self.ax = plt.subplots(figsize=(8,4))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        self.plot_feature(self.index)
        plt.show()
        
    def plot_feature(self, idx):
        self.ax.clear()
        
        feature = self.features[idx]
        self.ax.hist(self.df_signal[feature], bins=100, alpha=0.5, label='Signal', density=True, color='blue', edgecolor='white', linewidth=0.5)
        self.ax.hist(self.df_background[feature], bins=100, alpha=0.5, label='Background', density=True, color='pink', edgecolor='white', linewidth=0.5)
        
        self.ax.set_title(f'Histogram of {feature} ({idx+1}/{len(self.features)})')
        self.ax.set_xlabel(feature)
        self.ax.set_ylabel('Densité')
        self.ax.legend()
        self.ax.grid()
        
        self.fig.canvas.draw_idle()
        
    def on_key(self, event):
        if event.key == 'right':
            self.index = (self.index + 1) % len(self.features)
            self.plot_feature(self.index)
        elif event.key == 'left':
            self.index = (self.index - 1) % len(self.features)
            self.plot_feature(self.index)

# Usage
visualiseur = MiniVisualiseur(df_signal, df_background, features)
