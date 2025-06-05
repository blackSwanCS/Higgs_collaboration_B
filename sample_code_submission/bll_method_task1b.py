from iminuit import Minuit

# from iminuit.cost import ExtendedBinnedNLL
# from iminuit.cost import ExtendedUnbinnedNLL
# from resample import bootstrap
from scipy.stats import poisson

# from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
import math


def bll_method_1b(labels, scores, weights, N_bins=10):
    """Computes the estimated µ and 16&84-th quantiles by the Binned Likelihood Method

    Args:
        labels : 0 (bkg) or 1 (signal)
        scores : in [0,1]
        weights : weights of the events
        N_bins : number of bins in the histogram

    Returns:
        nothing (plots the negative log-likelihood obtained by the BLL and by the basic counting method)
    """
    # Initialisation
    n = len(scores)
    idx_list_S = []
    idx_list_B = []
    S_scores = []
    B_scores = []
    S_weights = []
    B_weights = []

    # Récupération des indices de la liste scores correspondant à Signal ou Bkg
    for k in range(n):
        if labels[k] == 1:
            idx_list_S.append(k)
        else:
            idx_list_B.append(k)
    for idx_S in idx_list_S:
        S_scores.append(scores[idx_S])
        S_weights.append(weights[idx_S])
    for idx_B in idx_list_B:
        B_scores.append(scores[idx_B])
        B_weights.append(weights[idx_B])

    # Construction de l'histogramme utilisé par la Binned Likelihood Method après
    S_hist = np.histogram(S_scores, bins=N_bins, range=(0, 1), weights=S_weights)
    B_hist = np.histogram(B_scores, bins=N_bins, range=(0, 1), weights=B_weights)

    # Ici nous pouvons plot l'histogramme des scores donné par BDT / NN ?

    x_bin_edges = np.linspace(0, 1, N_bins + 1)
    x = [x_bin_edges[k] for k in range(N_bins)]

    Si = S_hist[0]
    Ba = B_hist[0]

    plt.plot(x, Si, label="Signal")
    plt.plot(x, Ba, label="Background")
    plt.title("Signal and Background density distribution wrt the score")
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.show()

    S = np.sum(Si)
    B = np.sum(Ba)

    A1 = Si
    A2 = Ba

    # plt.plot(x, A1 + A2, label='n=S+B', linewidth=3)
    # plt.plot(x, A2, label='B', linestyle = '--')
    # plt.plot(x, A1, label='S', linestyle = ':')
    # plt.legend( facecolor='w')
    # plt.xlabel('variable')
    # plt.ylabel('Number of events')
    # plt.title('Signal and background event distributions')
    # plt.show()

    # Initializing the probability of an event (bkg or signal) to 0.
    pS = np.zeros([np.size(x_bin_edges) - 1, 1])
    pB = np.zeros([np.size(x_bin_edges) - 1, 1])
    for k in np.arange(0, np.size(x_bin_edges) - 1):
        pS[k] = (
            Si[k] / S
        )  # (number of signal in the k-th bin) / (total number of signal)
        pB[k] = Ba[k] / B  # (number of bkg in the k-th bin) / (total number of bkg)

    # And we draw the result of the bin contents:
    fig, fig_axes = plt.subplots(ncols=2, nrows=1)
    fig.set_size_inches(w=14, h=3)

    fig_axes[0].step(x_bin_edges[0:-1], pS, label="Signal")
    fig_axes[0].step(x_bin_edges[0:-1], pB, label="Background")
    fig_axes[0].set_title(r"${\bf Probability}$: bin content of Signal and Background")
    fig_axes[0].set_xlabel("X")
    fig_axes[0].set_ylabel("Probability")
    fig_axes[0].legend(facecolor="w")

    fig_axes[1].step(x_bin_edges[0:-1], S * pS + B * pB, label="S+B")
    fig_axes[1].step(x_bin_edges[0:-1], B * pB, label="Background")
    fig_axes[1].step(x_bin_edges[0:-1], S * pS, label="Signal")
    fig_axes[1].set_title(r"${\bf Counts}$: bin content of Signal and Background")
    fig_axes[1].set_xlabel("X")
    fig_axes[1].set_ylabel("Counts")
    fig_axes[1].legend(facecolor="w")

    plt.show()

    n = S + B
    nprim = np.round(n)

    # we are forced to round the values here otherwise we would get count numbers
    # which would not be integers. And this would be problematic with the Poisson
    # PMF below which is only defined for integer values for the data.
    y = np.round(Si + Ba)
    # y is an array : len(y) = nb of bins. y[k] is the total number of events in each bin

    # We define the bin content with the following function
    def BinContent(k, mu):
        return mu * S * pS[k] + B * pB[k]

    # We define the likelihood for a single bin"
    def likp(k, yk, mu):
        return poisson(BinContent(k, mu)).pmf(yk)

    # We define the full binned log-likelihood:
    def bll(mu):
        return -2 * sum([np.log(likp(k, y[k], mu)) for k in range(0, np.size(y))])

    # y[k] = s[k] + b[k]
    # BinContent(k,µ) = µ*s[k] + b[k]
    # likp(k,yk,µ) = Pr(Yk = yk) avec Yk suivant la loi de Poisson(µ*s[k] + b[k])
    # il n'y a donc qu'un seul µ, et on estime le µ qui "limite la casse"
    # sur un seul histogramme, µ peut être déterminé explicitement, mais dès qu'il y en a plusieurs il faut faire appel à un minimiseur
    # car la log-likelihood n'est plus si simple à étudier (somme de beaucoup de termes)
    # bll(µ) = -2 somme sur k des : ln Pr(Yk=y[k]) où y[k] est égal à s[k] + b[k] (arrondi à l'entier) et Yk suit la loi de Poisson(µ*s[k] + b[k])
    # c'est donc une NLL qui prend en compte tous les histogrammes. Pour la minimiser et trouver les quantiles à 50 +/- 68/2 = 50 +- 34 = 16 et 84 :
    # on fit (peut-être avec iMinuit, sinon avec polyfit) une parabole à cette NLL (cf. plot ci-dessus / ci-dessous : intersections de la parabole avec la cste à 1+min = intervalle 68% ie 1 sigma ie quantiles 16 et 84

    EPS = 0.0001  # trick to avoid potential division by zero during the minimization
    par_bnds = (EPS, None)  # Forbids parameter values to be negative, so mu>EPS here.
    par0 = 0.5  # quick bad guess to start with some value of mu...

    m = Minuit(bll, mu=par0)
    m.migrad()

    print("mu =", m.values["mu"])
    print("f(mu) =", m.fval)
    print("Erreur estimée sur mu =", m.errors["mu"])
    print("y  :  ", y)
    print("pB  :  ", pB)
    print("pS :  ", pS)
    print(np.sum(pB), np.sum(pS))
    print("B_hist", B_hist)
    print("S_hist", S_hist)

    ## Plot of the likelihoods

    mu_axis_values = np.linspace(0.5, 1.5, 10000)
    binned_loglike_values = np.array([bll(mu) for mu in mu_axis_values]).flatten()

    plt.plot(
        mu_axis_values,
        binned_loglike_values - min(binned_loglike_values),
        label="binned log-likelihood",
    )
    plt.hlines(
        1, min(mu_axis_values), max(mu_axis_values), linestyle="--", color="tab:gray"
    )

    idx = np.argwhere(
        np.diff(np.sign(binned_loglike_values - min(binned_loglike_values) - 1))
    ).flatten()
    print("an equation has been solved !")
    plt.xlabel(r"$\mu$")
    plt.ylabel(r"$-2\log {\cal L}(\mu) + 2\log {\cal L}_{\rm min}$")
    plt.title(r"Log-likelihood profile with respect to $\mu$ using")

    plt.plot(mu_axis_values[idx], [1, 1], "ko", label=r"$1\sigma$ interval")
    plt.plot(mu_axis_values[idx[0]] * np.ones(2), [0, 1], "k--")
    plt.plot(mu_axis_values[idx[1]] * np.ones(2), [0, 1], "k--")
    delta_mu = np.diff(mu_axis_values[idx])
    sigma_mu = delta_mu / 2
    plt.plot(
        mu_axis_values,
        ((mu_axis_values - 1) / sigma_mu) ** 2,
        linestyle="-.",
        color="tab:gray",
        label="parabola approx.",
    )

    # redo the basic counting log-likelihood
    def loglik(mu, n):
        eps = 0.000001
        return -2 * np.log(poisson.pmf(n, mu * S + B) + eps)

    loglike_values = np.array([loglik(mu, nprim) for mu in mu_axis_values])
    plt.plot(
        mu_axis_values,
        loglike_values - min(loglike_values),
        label="count log-likelihood",
    )
    plt.legend(facecolor="w")
    # Defining the binned log-likelihood best fit mu parameter:
    muhat = mu_axis_values[np.argmin(binned_loglike_values)]
    # Préparation du texte à afficher sur le graphique
    textstr = "\n".join(
        (
            r"$\hat{\mu} = %.3f$" % muhat,
            r"$16^{\rm th}$ quantile: %.3f" % mu_axis_values[idx[0]],
            r"$84^{\rm th}$ quantile: %.3f" % mu_axis_values[idx[1]],
            r"$\delta_{\mu} \approx %.3f$" % delta_mu,
        )
    )

    # Positionner le texte sur le graphe
    props = dict(boxstyle="round", facecolor="white", alpha=0.8)
    plt.text(
        0.5,
        0.95,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )
    plt.show()

    # Printing the output of the fit:
    print("Shape analysis. Best fit parameter and uncertainty:\n")
    print("Estimated µ with BLL / Minuit : ", muhat)
    print("16-th quantile : ", abs(mu_axis_values[idx[0]]))
    print("84-th quantile : ", mu_axis_values[idx[1]])

    return 1
