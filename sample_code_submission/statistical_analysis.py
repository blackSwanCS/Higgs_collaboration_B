import numpy as np
from HiggsML.systematics import systematics
from scipy import stats
from iminuit import Minuit
import pandas as pd
import matplotlib.pyplot as plt
from bll_method_task1b import bll_method_1b
from bll_method_task2 import bll_method_2

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
    Perform calculations to calculate mu
    Dummy code, replace with actual calculations
    Feel free to add more functions and change the function parameters
    """

    score = score.flatten() > 0.9
    mu = (np.sum(score * weight) - saved_info["beta"]) / saved_info["gamma"]
    del_mu_stat = (
        np.sqrt(saved_info["beta"] + saved_info["gamma"]) / saved_info["gamma"]
    )
    del_mu_sys = abs(0.0 * mu)
    del_mu_tot = np.sqrt(del_mu_stat**2 + del_mu_sys**2)

    return {
        "mu_hat": mu,
        "del_mu_stat": del_mu_stat,
        "del_mu_sys": del_mu_sys,
        "del_mu_tot": del_mu_tot,
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

    # Formula de la log-vraisemblance négative binnie étendue
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
    plt.hist(
        score_background,
        bins=bins,
        weights=weights_background,
        alpha=0.6,
        label="Background",
        color="skyblue",
        density=True,
    )
    plt.hist(
        score_signal,
        bins=bins,
        weights=weights_signal,
        alpha=0.6,
        label="Signal",
        color="orange",
        density=True,
    )

    # Ligne verticale pour le seuil courant (0.5)
    plt.axvline(0.5, color="red", linestyle="--", label="Seuil = 0.5")

    plt.xlabel("Score du modèle")
    plt.ylabel("Distribution normalisée")
    plt.title("Distribution des scores du modèle pour signal et background")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def compute_ams(s, b):
    if b <= 0:
        return 0
    return np.sqrt(2 * ((s + b) * np.log(1 + s / b) - s))

def calculate_saved_info(model, holdout_set):
    """
    Calculate the saved_info dictionary for mu calculation
    Replace with actual calculations
    """
    
    score = model.predict(holdout_set["data"])

    from systematic_analysis import tes_fitter
    from systematic_analysis import jes_fitter
    # Execution Tache 1A
    # compute_threshold = calculate_best_threshold(score,holdout_set)

    score = model.predict(holdout_set["data"])
    label = holdout_set["labels"]
    weights = holdout_set["weights"]
    # Execution tache 1B
    task_1B = bll_method_2(model,holdout_score,label,score,weights)
    score = score.flatten() > 0.9
    score = score.astype(int)
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

def calculate_best_threshold(score,holdout_set):
    n = 95
    ams = [i/100 for i in range (n)]
    print("score shape before threshold", score.shape)
    for i in range (n) :
        copy_score = score.copy()
        copy_score = copy_score.flatten() > i/100
        copy_score = copy_score.astype(int)
        label = holdout_set["labels"]
        gamma = np.sum(holdout_set["weights"] * copy_score * label)

        beta = np.sum(holdout_set["weights"] * copy_score * (1 - label))

        ams[i] = compute_ams(gamma,beta)

    t = [i/100 for i in range (n)]
    plt.figure(figsize=(8, 5))
    plt.plot(t, ams, marker='o')
    plt.xlabel("Seuil (Threshold)")
    plt.ylabel("AMS")
    plt.title("AMS en fonction du Threshold")
    plt.grid(True)
    plt.show()
    ams_max = max(ams)
    threshold_max = t[ams.index(ams_max)]
    print("ams_max:",ams_max," for a threshold_max:",threshold_max)
    return 1



#### Task 2
def nll_with_systematics(mu, tes, jes, saved_info, hist_data):
    """
    NLL avec mu, tes, jes. 
    Utilise les fits système pour ajuster gamma et beta.
    """
    gamma_func = saved_info["tes_fit"]
    beta_func = saved_info["jes_fit"]

    # Évalue les fonctions fit pour obtenir gamma et beta modifiés
    gamma = gamma_func(tes)
    beta = beta_func(jes)

    score = hist_data["score"].flatten() > saved_info.get("threshold", 0.5)
    weight = hist_data["weights"]
    selected = score.astype(int)

    # Compte observé
    obs = np.sum(selected * weight)

    # Espérance : mu * gamma + beta
    expected = mu * gamma + beta

    # NLL de Poisson
    if expected <= 0:
        return 1e6
    return expected - obs * np.log(expected)




def compute_mu_with_systematics(score, weight, saved_info):
    """
    Minimise la NLL avec TES et JES.
    """

    data = {
        "score": score,
        "weights": weight
    }

    def wrapped_nll(mu, tes, jes):
        return nll_with_systematics(mu, tes, jes, saved_info, data)

    # Minuit
    minuit = Minuit(wrapped_nll, mu=1.0, tes=0.0, jes=0.0)
    minuit.errordef = Minuit.LIKELIHOOD
    minuit.limits = {
        "mu": (0, 5),
        "tes": (-5, 5),
        "jes": (-5, 5)
    }

    result = minuit.migrad()
    mu_hat = minuit.values["mu"]
    mu_err = minuit.errors["mu"]

    return {
        "mu_hat": mu_hat,
        "del_mu_stat": 0.0,
        "del_mu_sys": mu_err,
        "del_mu_tot": mu_err,
    }