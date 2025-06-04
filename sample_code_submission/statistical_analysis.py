import numpy as np
from HiggsML.systematics import systematics
from scipy import stats
from iminuit import Minuit
import pandas as pd
import matplotlib.pyplot as plt


"""
Task 1a : Counting Estimator
1.write the saved_info dictionary such that it contains the following keys
    1. beta
    2. gamma
2. Estimate the mu using the formula
    mu = (sum(score * weight) - beta) / gamma
3. return the mu and its uncertainty

Task 1b : Stat-Only Likelihood Estimator
1. Modify the estimation of mu such that it uses the likelihood function
    1. Write a function for the likelihood function which profiles over mu
    2. Use Minuit to minimize the NLL

Task 2 : Systematic Uncertainty
1. substitute the beta and gamma with the tes_fit and jes_fit functions
2. Write a function to likelihood function which profiles over mu, tes and jes
3. Use Minuit to minimize the NLL
4. return the mu and its uncertainty
"""


def compute_mu(score, weight, saved_info):
    """
    Estimation de mu avec profilage sur TES et JES (syst. incertitudes).
    Utilise des polynômes ajustés bin-par-bin sur les histogrammes.

    Retourne :
        - mu_hat : valeur optimale
        - del_mu_tot : incertitude totale (stat + syst)
    """
    from iminuit import Minuit
    import numpy as np

    # Aplatir les tableaux proprement
    score = np.asarray(score).ravel()
    weight = np.asarray(weight).ravel()

    # Histogramme observé pondéré des prédictions
    obs_counts, bin_edges = np.histogram(score, bins=100, range=(0, 1), weights=weight)

    # Extraction des fonctions TES/JES (chacune est un tuple : (fit_params, eval_func))
    tes_fit_params, tes_eval_func = saved_info["tes_fit"]
    jes_fit_params, jes_eval_func = saved_info["jes_fit"]

    # Valeurs nominales des normalisations signal et bruit de fond
    ns_nom = saved_info["gamma"]
    nb_nom = saved_info["beta"]

    # Paramètres fixes pour les formes de distributions
    sigma = 0.1   # pour le signal
    lambd = 0.2   # pour le bruit de fond

    # Fonction de log-vraisemblance négative à profiler
    def nll(mu, tes, jes):
        # Appliquer les fonctions TES/JES aux polynômes par bin
        ns_bins = np.array(tes_eval_func(tes, tes_fit_params))  # bins signal
        nb_bins = np.array(jes_eval_func(jes, jes_fit_params))  # bins fond

        # Comptes attendus dans chaque bin
        expected_counts = mu * ns_bins + nb_bins

        # Sécurité numérique
        expected_counts = np.clip(expected_counts, 1e-9, None)

        # NLL binnie (Poisson)
        return np.sum(expected_counts - obs_counts * np.log(expected_counts))

    # Optimisation avec Minuit
    minuit = Minuit(nll, mu=1.0, tes=1.0, jes=1.0)
    minuit.limits["mu"] = (0, 5)
    minuit.limits["tes"] = (0.9, 1.1)
    minuit.limits["jes"] = (0.9, 1.1)
    minuit.errordef = Minuit.LIKELIHOOD

    minuit.migrad()

    # Résultats
    mu_hat = minuit.values["mu"]
    del_mu_tot = minuit.errors["mu"]

    return {
        "mu_hat": mu_hat,
        "del_mu_stat": None,       # séparé uniquement si NLL stat-only disponible
        "del_mu_sys": None,
        "del_mu_tot": del_mu_tot,  # total = stat + syst dans cette config
    }


def signal(xe, ns, mu, sigma):
    return ns * stats.norm(mu, sigma).cdf(xe)

def background(xe, nb, lambd):
    return nb * stats.expon(0.0, lambd).cdf(xe)

def total(xe, ns, mu, sigma, nb, lambd):
    return signal(xe, ns, mu, sigma) + background(xe, nb, lambd)

def extended_binned_nll(obs_counts, bin_edges, ns, mu, sigma, nb, lambd):
    # Calcule les comptes cumulés attendus par bin
    xe = bin_edges
    expected_cdf = total(xe, ns, mu, sigma, nb, lambd)

    # Comptes attendus dans chaque bin = différence entre les valeurs CDF successives
    expected_counts = np.diff(expected_cdf)

    # Sert à éviter log(0)
    expected_counts = np.clip(expected_counts, 1e-9, None)

    # Formule de la log-vraisemblance négative binnie étendue
    nll = np.sum(expected_counts - obs_counts * np.log(expected_counts))
    return nll


def plot_score_distributions(score, labels, weights=None, bins=50):
    """
    Affiche la distribution des scores pour le signal et le bruit de fond.
    Paramètres :
    - score : array (probabilité prédite par le modèle)
    - labels : array (0 = background, 1 = signal)
    - weights : array facultatif (poids des événements)
    - bins : nombre de bins de l'histogramme
    """
    score = score.flatten()
    labels = labels.flatten()

    # Séparer signal et bruit
    score_signal = score[labels == 1]
    score_background = score[labels == 0]
    
    weights_signal = weights[labels == 1] if weights is not None else None
    weights_background = weights[labels == 0] if weights is not None else None

    # Tracer les histogrammes
    plt.figure(figsize=(10, 6))
    plt.hist(score_background, bins=bins, weights=weights_background,
             alpha=0.6, label='Background', color='skyblue', density=True)
    plt.hist(score_signal, bins=bins, weights=weights_signal,
             alpha=0.6, label='Signal', color='orange', density=True)

    # Ligne verticale pour le seuil courant (0.5)
    plt.axvline(0.5, color='red', linestyle='--', label='Seuil = 0.5')

    plt.xlabel("Score du modèle")
    plt.ylabel("Distribution normalisée")
    plt.title("Distribution des scores du modèle pour signal et background")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def calculate_saved_info(model, holdout_set):
    """
    Calculate the saved_info dictionary for mu calculation
    Replace with actual calculations
    """

    score = model.predict(holdout_set["data"])

    from systematic_analysis import tes_fitter
    from systematic_analysis import jes_fitter

    print("score shape before threshold", score.shape)
    # Execution Tache 1A
    compute_threshold = calculate_best_threshold(score,holdout_set)

    score = score.flatten() > 0.5
    score = score.astype(int)

    label = holdout_set["labels"]
    # Execution tache 1B
    #task_1B = bll_method(label,score)
    score = score.flatten() > 0.9
    score = score.astype(int)

    print("score shape after threshold", score.shape)

    gamma = np.sum(holdout_set["weights"] * score * label)

    beta = np.sum(holdout_set["weights"] * score * (1 - label))

    saved_info = {
        "beta": beta,
        "gamma": gamma,
        "tes_fit": tes_fitter(model, holdout_set),
        "jes_fit": jes_fitter(model, holdout_set),
    }

    print("saved_info", saved_info)

    return saved_info


def compute_ams(s, b):
    if b <= 0:
        return 0
    return np.sqrt(2 * ((s + b) * np.log(1 + s / b) - s))


def scan_threshold_ams(score, labels, weights, plot=True):
    min_score = np.min(score)
    max_score = np.max(score)
    thresholds = np.linspace(min_score + 1e-4, max_score - 1e-4, 100)

    best_threshold = 0.5
    best_ams = 0

    ams_values = []
    for thresh in thresholds:
        selected = score.flatten() >= thresh
        s = np.sum(weights[selected & (labels == 1)])
        b = np.sum(weights[selected & (labels == 0)])
        ams = compute_ams(s, b)
        ams_values.append(ams)
        if ams > best_ams:
            best_ams = ams
            best_threshold = thresh

    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(thresholds, ams_values, label="AMS vs Threshold")
        plt.axvline(best_threshold, color='r', linestyle='--', label=f"Best Threshold = {best_threshold:.3f}")
        plt.xlabel("Threshold")
        plt.ylabel("AMS")
        plt.title("Scan of AMS as a Function of Score Threshold")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return best_threshold, best_ams
