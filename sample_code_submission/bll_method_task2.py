from iminuit import Minuit
#from iminuit.cost import ExtendedBinnedNLL
#from iminuit.cost import ExtendedUnbinnedNLL
#from resample import bootstrap
from scipy.stats import poisson
#from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
import math
from copy_systematic_analysis import tes_fitter , jes_fitter


def bll_method_2(model,holdout_set,labels, scores, weights, N_bins = 10):
    jes = jes_fitter(model,holdout_set)
    tes = jes_fitter(model,holdout_set)
    #Initialisation
    n = len(scores)
    idx_list_S = []
    idx_list_B = []
    S_scores = []
    B_scores = []
    S_weights = []
    B_weights = []
    #Récupération des indices de la liste scores correspondant à Signal ou Bkg
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

    #Construction de l'histogramme utilisé par la Binned Likelihood Method après
    S_hist = np.histogram(S_scores,bins = N_bins, range=(0,1), weights= S_weights)
    B_hist = np.histogram(B_scores,bins = N_bins,range=(0,1), weights= B_weights)

    # Ici nous pouvons plot l'histogramme des scores donné par BDT / NN
    # Y : array de population dans les bins (donc len(Y) = len(x))
    # x : array des scores délimitant les bins
    # Il faut extraire l'histogramme qui contient les labels, scores, et densités

    x_bin_edges = np.linspace(0, 1, N_bins+1)
    x = [x_bin_edges[k] for k in range(N_bins)]

    Si = S_hist[0]
    Ba = B_hist[0]

    plt.plot(x, Si, label= 'Signal')
    plt.plot(x, Ba, label='Background')
    plt.title('Signal and Background density distribution wrt the score')
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.show()

    # Plotting a typical signal with a large background
    S = np.sum(Si)
    B = np.sum(Ba)

    A1 = Si
    A2 = Ba

    #plt.plot(x, A1 + A2, label='n=S+B', linewidth=3)
    #plt.plot(x, A2, label='B', linestyle = '--')
    #plt.plot(x, A1, label='S', linestyle = ':')
    #plt.legend( facecolor='w')
    #plt.xlabel('variable')
    #plt.ylabel('Number of events')
    #plt.title('Signal and background event distributions')
    #plt.show()

    # We have to fix the binning (interval : [0,1], so just choose the number of bins)

    # We initialize the probability of an event being a signal or background one to 0.
    pS = np.zeros([np.size(x_bin_edges)-1, 1])
    pB = np.zeros([np.size(x_bin_edges)-1, 1])
    for k in np.arange(0, np.size(x_bin_edges)-1):
        pS[k] = Si[k]/S # (number of signal in the k-th bin) / (total number of signal)
        pB[k] = Ba[k]/B # (number of bkg in the k-th bin) / (total number of bkg)


    # And we draw the result of the bin contents:
    fig, fig_axes = plt.subplots(ncols=2, nrows=1)
    fig.set_size_inches(w=14, h=3)

    fig_axes[0].step(x_bin_edges[0:-1], pS, label='Signal')
    fig_axes[0].step(x_bin_edges[0:-1], pB, label='Background')
    fig_axes[0].set_title(r'${\bf Probability}$: bin content of Signal and Background')
    fig_axes[0].set_xlabel('X')
    fig_axes[0].set_ylabel('Probability')
    fig_axes[0].legend(facecolor= 'w')

    fig_axes[1].step(x_bin_edges[0:-1], S*pS+B*pB, label='S+B')
    fig_axes[1].step(x_bin_edges[0:-1], B*pB, label='Background')
    fig_axes[1].step(x_bin_edges[0:-1], S*pS, label='Signal')
    fig_axes[1].set_title(r'${\bf Counts}$: bin content of Signal and Background')
    fig_axes[1].set_xlabel('X')
    fig_axes[1].set_ylabel('Counts')
    fig_axes[1].legend(facecolor= 'w')

    plt.show()

    n = S+B
    nprim = np.round(n)

    # we are forced to round the values here otherwise we would get count numbers
    # which would not be integers. And this would be problematic with the Poisson
    # PMF below which is only defined for integer values for the data.
   
    # array : len(y) = nb of bins. y[k] is the total number of events in each bin

    # We define the bin content with the following function
    y = np.round(S*pS + B*pB)

    def parabola(fitter_i,alpha):
        if len(fitter_i) != 3:
            print("pas une parabole")
        return fitter_i[0]*(alpha**2) + fitter_i[1]*alpha + fitter_i[0]
    
    def droite(fitter_i,alpha):
        if len(fitter_i) != 2:
            print("pas une droite")
        return fitter_i[0]*alpha + fitter_i[1]
        
    
    def gamma(model,holdout_set,bin_idx,alpha_tes,alpha_jes):
        gamma_alpha_jes = parabola(jes[2][bin_idx],alpha_jes)
        gamma_alpha_tes = parabola(tes[2][bin_idx],alpha_tes)
        return gamma_alpha_jes + gamma_alpha_tes

    def beta(model,holdout_set,bin_idx,alpha_tes,alpha_jes):
        beta_alpha_jes = parabola(jes[3][bin_idx],alpha_jes)
        beta_alpha_tes = parabola(tes[3][bin_idx],alpha_tes)
        return beta_alpha_jes + beta_alpha_tes

    def BinContent(bin_idx, mu,alpha_tes,alpha_jes,model,holdout_set):
        return mu*gamma(model,holdout_set,bin_idx,alpha_tes,alpha_jes)*pS[bin_idx]+beta(model,holdout_set,bin_idx,alpha_tes,alpha_jes)*pB[bin_idx]

    # We define the likelihood for a single bin"
    def likp(bin_idx, yk, mu,alpha_tes,alpha_jes,model,holdout_set):
        eps = 1e-12
        proba = poisson(BinContent(bin_idx, mu,alpha_tes,alpha_jes,model,holdout_set)).pmf(yk)
        print("proba",proba)
        if proba == 0:
            return eps
        else:
            return proba

    # We define the full binned log-likelihood:
    def bll(mu,alpha_tes,alpha_jes,model,holdout_set):
        sigma0 = 0.01
        alpha0 = 1
        return -2 * sum([np.log(likp(bin_idx, y[bin_idx], mu,alpha_tes,alpha_jes,model,holdout_set)) for bin_idx in range(N_bins)]) + ((alpha_jes - alpha0) / sigma0)**2 + ((alpha_tes - alpha0) / sigma0)**2

    EPS = 0.0001 # trick to avoid potential division by zero during the minimization
    par_bnds = ((EPS, None)) # Forbids parameter values to be negative, so mu>EPS here.
    par0 = 0.5 # quick bad guess to start with some value of mu...

    def make_bll(model, holdout_model):
        def wrapped(mu, alpha_jes, alpha_tes):
            return bll(mu, alpha_jes, alpha_tes, model, holdout_model)
        return wrapped 

    my_bll = make_bll(model, holdout_set)

    m = Minuit(my_bll, mu=0.5, alpha_tes=0.9, alpha_jes=0.9)
    m.limits["mu"] = (0, 5)
    m.limits["alpha_tes"] = (0.8, 1.2)
    m.limits["alpha_jes"] = (0.8, 1.2)
    m.migrad()

    print("mu =", m.values["mu"])
    print("mualpha_jes =", m.values["alpha_jes"])
    print("alpha_tes =", m.values["alpha_tes"])

    print("f(mu) =", m.fval)
    print("Erreur estimée sur mu =", m.errors["mu"])
    print("y  :  ",y)
    print("pB  :  ",pB)
    print("pS :  ", pS)
    print(np.sum(pB),np.sum(pS))
    print("B_hist" , B_hist)
    print("S_hist" , S_hist)

    ## Plot of the likelihoods

    mu_axis_values = np.linspace(0.5, 1.5, 100)
    binned_loglike_values = np.array([my_bll(mu,m.values["alpha_tes"],m.values["alpha_jes"]) for mu in mu_axis_values]).flatten()

    plt.plot(mu_axis_values, binned_loglike_values - min(binned_loglike_values),
            label='binned log-likelihood')
    plt.hlines(1, min(mu_axis_values), max(mu_axis_values), linestyle= '--', color= 'tab:gray')

    idx = np.argwhere(np.diff(np.sign(
            binned_loglike_values - min(binned_loglike_values) - 1
            ))).flatten()

    plt.xlabel(r'$\mu$')
    plt.ylabel(r'$-2\log {\cal L}(\mu) + 2\log {\cal L}_{\rm min}$')
    plt.title(r'Log-likelihood profile with respect to $\mu$')

    plt.plot(mu_axis_values[idx], [1, 1], 'ko', label=r'$1\sigma$ interval')
    plt.plot(mu_axis_values[idx[0]]*np.ones(2), [0, 1], 'k--')
    plt.plot(mu_axis_values[idx[1]]*np.ones(2), [0, 1], 'k--')
    sigma_mu = np.diff(mu_axis_values[idx])/2
    plt.plot(mu_axis_values, ((mu_axis_values-1)/sigma_mu)**2, linestyle='-.',
            color= 'tab:gray', label='parabola approx.')

    # redo the basic counting log-likelihood
    def loglik(mu, n):
        eps = 0.000001
        return -2*np.log(poisson.pmf(n, mu*S + B)+eps)
    loglike_values = np.array([loglik(mu, nprim) for mu in mu_axis_values])
    plt.plot(mu_axis_values, loglike_values - min(loglike_values),
            label='count log-likelihood')
    plt.legend(facecolor = 'w')
    plt.show()
    # Defining the binned log-likelihood best fit mu parameter:
    muhat = mu_axis_values[np.argmin(binned_loglike_values)]

    # Printing the output of the fit:
    print('Shape analysis. Best fit parameter and uncertainty:\n')
    print("Estimated µ with BLL / Minuit : " , muhat)
    print("16-th quantile : ", abs(mu_axis_values[idx[0]]))
    print("84-th quantile : " , mu_axis_values[idx[1]])

    return 1
    
##Test avec des données de forme analogue aux histogrammes rencontrés
"""
def decreasing_distribution(n):
    x = np.linspace(0, 1, n)
    return np.exp(-5 * x)  # décroissance exponentielle rapide

# Fonction pour générer une distribution rapidement croissante
def increasing_distribution(n):
    x = np.linspace(0, 1, n)
    return 1 - np.exp(-5 * x)  # croissance rapide vers 1

# Générer les distributions
dist_decroissante = decreasing_distribution(1000)
dist_croissante = increasing_distribution(60)

# Normaliser pour que les valeurs soient entre 0 et 1
dist_decroissante /= dist_decroissante.max()
dist_croissante /= dist_croissante.max()

dist_decroissante[0] -= 0.001
dist_croissante[-1] -= 0.001
# Affichage
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(dist_decroissante)
plt.title("Décroissante rapide (1000 valeurs)")
plt.xlabel("Index")
plt.ylabel("Valeur")

plt.subplot(1, 2, 2)
plt.plot(dist_croissante)
plt.title("Croissante rapide (60 valeurs)")
plt.xlabel("Index")

plt.tight_layout()
plt.show()

Scores = np.concatenate((dist_croissante,dist_decroissante))
Lab = [1 for _ in range(60)] + [0 for _ in range(1000)]

bll_method(Lab,Scores)
"""