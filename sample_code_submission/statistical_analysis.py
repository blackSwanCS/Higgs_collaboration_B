import numpy as np
from HiggsML.systematics import systematics
from scipy import stats
from iminuit import Minuit

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

    score = score.flatten() > 0.5
    score = score.astype(int)

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

    # Formule de la log-vraisemblance négative binnie étendue
    nll = np.sum(expected_counts - obs_counts * np.log(expected_counts))
    return nll



def calculate_saved_info(model, holdout_set):
    """
    Calculate the saved_info dictionary for mu calculation
    Replace with actual calculations
    """

    score = model.predict(holdout_set["data"])

    from systematic_analysis import tes_fitter
    from systematic_analysis import jes_fitter

    print("score shape before threshold", score.shape)

    score = score.flatten() > 0.5
    score = score.astype(int)

    label = holdout_set["labels"]

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
