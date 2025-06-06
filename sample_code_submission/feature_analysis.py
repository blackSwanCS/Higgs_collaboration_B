import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_jet_correlation_matrices(df, jet_column='PRI_n_jets', show_annotations=True):
    """
    Affiche les matrices de corrélation PRI vs DER pour 0, 1, et ≥2 jets.

    Paramètres :
        df (pd.DataFrame) : Le DataFrame contenant les données.
        jet_column (str) : Le nom de la colonne contenant le nombre de jets (par défaut 'PRI_n_jets').
        show_annotations (bool) : Affiche les valeurs numériques dans la heatmap si True.
    """
    # Séparation des datasets
    df_dict = {
        '0 jets': df[df[jet_column] == 0].copy(),
        '1 jet': df[df[jet_column] == 1].copy(),
        '2 jets et plus': df[df[jet_column] >= 2].copy()
    }

    # Pour chaque sous-dataset
    for title, df_sub in df_dict.items():
        features_PRI = [col for col in df_sub.columns if col.startswith('PRI_')]
        features_DER = [col for col in df_sub.columns if col.startswith('DER_')]
        features = features_PRI + features_DER

        corr_matrix = df_sub[features].corr()
        corr_PRI_DER = corr_matrix.loc[features_PRI, features_DER]

        # Affichage
        plt.figure(figsize=(14, 8))
        sns.heatmap(corr_PRI_DER, cmap='coolwarm', center=0, annot=show_annotations, fmt='.2f')
        plt.title(f'Corrélation PRI vs DER - {title}')
        plt.tight_layout()
        plt.show()

# Exemple d'utilisation :
file_path = '/Users/raph/Desktop/Higgs_collaboration_B/blackSwan_data/blackSwan_data.parquet'
df = pd.read_parquet(file_path)

# Appel de la fonction
plot_jet_correlation_matrices(df)


def systematics_dependence(data):
    pass


def minimal_dependent_features(data):
    return data.columns
