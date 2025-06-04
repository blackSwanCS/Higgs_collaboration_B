import numpy as np
from HiggsML.systematics import systematics
import matplotlib.pyplot as plt
import numpy as np
import os


def tes_fitter(
    model,
    train_set,
):
    """
    Task 1 : Analysis TES Uncertainty
    1. Loop over different values of tes and make store the score
    2. Make a histogram of the score

    Task 2 : Fit the histogram
    1. Write a function to loop over different values of tes and histogram and make fit function for each bin in the histogram
    2. store the fit functions in an array
    3. return the fit functions

      histogram and make fit function which transforms the histogram for any given TES

    """

    bins = 25
    # bin_indices = [5 * i for i in range(20)] + [99]  # Indices of bins to analyze
    bin_indices = [0, 4, 9, 14, 19, 24]  # Indices of bins to analyze

    syst_set = systematics(train_set, tes=1)

    target = syst_set["labels"]
    signal_field = syst_set["data"][target == 1]
    background_field = syst_set["data"][target == 0]

    score_signal = model.predict(signal_field)
    signal_weights = syst_set["weights"][target == 1]
    histogram_nominal_signal, _ = np.histogram(
        score_signal, bins=bins, range=(0, 1), weights=signal_weights
    )

    score_background = model.predict(background_field)
    background_weights = syst_set["weights"][target == 0]
    histogram_nominal_background, _ = np.histogram(
        score_background, bins=bins, range=(0, 1), weights=background_weights
    )

    def fit_function(array, maxi=2):
        """tes est toujours défini entre 0.9 et 1.1
        array correspond à la liste des valeurs prise par l'histogramme après isolement d'un bin
        """
        meilleur = 1
        R_meilleur = np.inf
        R = 0
        for deg in range(0, maxi + 1):
            parameters = np.polyfit(tes_range, array, deg)
            # print(f"{deg} : {parameters}")
            for ind in range(len(tes_range)):
                y = 0
                for i in range(deg + 1):
                    y += parameters[i] * tes_range[ind] ** (deg - i)
                R += (y - array[ind]) ** 2
            if R < R_meilleur:
                meilleur = deg
                R_meilleur = R
            R = 0
        # print("meilleur :", meilleur)

        return np.polyfit(tes_range, array, meilleur)

    show_background = False  # Set to True if you want to show background in the plots

    for bin_index in bin_indices:
        
        bin_index_graph = bin_index+1

        first_bin_nominal_signal = histogram_nominal_signal[bin_index]
        first_bin_nominal_background = histogram_nominal_background[bin_index]

        delta_S_signal = []
        delta_S_background = []

        tes_range = np.linspace(0.9, 1.1, 21)
        for idx_tes in tes_range:
            # Signal
            syst_set = systematics(train_set, tes = idx_tes)
            target = syst_set["labels"]
            signal_field = syst_set["data"][target == 1]
            background_field = syst_set["data"][target == 0]
            score_signal = model.predict(signal_field)
            weights_signal = syst_set["weights"][target == 1]
            histogram_signal, _ = np.histogram(
                score_signal, bins=bins, range=(0, 1), weights=weights_signal
            )

            # Background
            score_background = model.predict(background_field)
            weights_background = syst_set["weights"][target == 0]
            histogram_background, _ = np.histogram(
                score_background, bins=bins, range=(0, 1), weights=weights_background
            )

            first_bin_signal = histogram_signal[bin_index]
            first_bin_background = histogram_background[bin_index]
            delta_signal = first_bin_signal - first_bin_nominal_signal
            delta_background = first_bin_background - first_bin_nominal_background

            delta_S_signal.append(delta_signal)
            delta_S_background.append(delta_background)

        plt.figure(figsize=(20, 10))
        plt.scatter(tes_range, delta_S_signal, label="Signal", color="blue")
        if show_background:
            plt.scatter(tes_range, delta_S_background, label='Background', color="orange")
        plt.xlabel("TES")
        plt.ylabel(r"$\Delta\ N$")
        plt.title(f"Shifted bin no. {bin_index_graph} of the Histogram")

        # Fit polynomial to delta_S_signal and delta_S_background
        fit_params_signal = fit_function(delta_S_signal)
        fit_params_background = fit_function(delta_S_background)

        # Generate smooth TES values for plotting the fit
        tes_smooth = np.linspace(0.9, 1.1, 100)
        fit_curve_signal = np.polyval(fit_params_signal, tes_smooth)
        fit_curve_background = np.polyval(fit_params_background, tes_smooth)

        # Plot the fit curves on top of the scatter plot
        plt.plot(tes_smooth, fit_curve_signal, label="Signal fit", color="blue", linestyle="--")
        if show_background:
            plt.plot(tes_smooth, fit_curve_background, label="Background fit", color="orange", linestyle="--")
        plt.legend()
        plt.grid()
        plt.title(f"Shifted bin no. {bin_index_graph} of the Histogram")
        os.makedirs("bin_graphs", exist_ok=True)
        if show_background:
            plt.savefig(f"bin_graphs/tes_analysis_bin_{bin_index_graph}_with_bg.png")
        else:
            plt.savefig(f"bin_graphs/tes_analysis_bin_{bin_index_graph}.png")
        plt.close()


    ######## Deux fonctions à regrouper
    ##### Il faudra appeler fit_function pour tous les bins à étudier et stocker les paramètres renvoyés dans une liste (le résultat est une liste de liste prise comme l'argument fitting_pol pour la 2de fonction)

    def eval_alpha(alpha, delta_S_signal, delta_S_background, maxi=3):
        """input :
        alpha -> the estimation of the parameter wanted

        fitting_pol -> list of the polynoms which fit the plots of Delta S as a function of the parameter alpha (Highest degree last) the signal in first, the background after

        output :
        list of S + Delta S for each bin"""


        fitting_pol = []
        for i in range(len(delta_S_signal)):
            fitting_pol.append(fit_function(delta_S_signal[i]), maxi=maxi)

        for i in range(len(delta_S_background)):
            fitting_pol.append(fit_function(delta_S_background[i]), maxi=maxi)

        list_S_plus_delta_S = []
        for i in range(len(fitting_pol)):
            list_S_plus_delta_S.append(np.polyval(fitting_pol[i][::-1], alpha))
        return list_S_plus_delta_S

    return fit_function, eval_alpha

def jes_fitter(
    model,
    train_set,
):
    """
    Task 1 : Analysis JES Uncertainty
    1. Loop over different values of jes and store the score
    2. Make a histogram of the score
 
    Task 2 : Fit the histogram
    1. Write a function to loop over different values of JES and histogram and make fit function for each bin in the histogram
    2. store the fit functions in an array
    3. return the fit functions
 
      histogram and make fit function which transforms the histogram for any given jes
 
    """
    bins = 25
    # bin_indices = [5 * i for i in range(20)] + [99]  # Indices of bins to analyze
    bin_indices = [0, 4, 9, 14, 19, 24]  # Indices of bins to analyze
 
    syst_set = systematics(train_set, jes=1)
 
    target = syst_set["labels"]
    signal_field = syst_set["data"][target == 1]
    background_field = syst_set["data"][target == 0]
 
    score_signal = model.predict(signal_field)
    signal_weights = syst_set["weights"][target == 1]
    histogram_nominal_signal, _ = np.histogram(
        score_signal, bins=bins, range=(0, 1), weights=signal_weights
    )
 
    score_background = model.predict(background_field)
    background_weights = syst_set["weights"][target == 0]
    histogram_nominal_background, _ = np.histogram(
        score_background, bins=bins, range=(0, 1), weights=background_weights
    )
 
    def fit_function(array, maxi=2):
        """jes est toujours défini entre 0.9 et 1.1
        array correspond à la liste des valeurs prise par l'histogramme après isolement d'un bin
        """
        meilleur = 1
        R_meilleur = np.inf
        R = 0
        for deg in range(0, maxi + 1):
            parameters = np.polyfit(jes_range, array, deg)
            # print(f"{deg} : {parameters}")
            for ind in range(len(jes_range)):
                y = 0
                for i in range(deg + 1):
                    y += parameters[i] * jes_range[ind] ** (deg - i)
                R += (y - array[ind]) ** 2
            if R < R_meilleur:
                meilleur = deg
                R_meilleur = R
            R = 0
        # print("meilleur :", meilleur)
 
        return np.polyfit(jes_range, array, meilleur)
 
    show_background = False  # Set to True if you want to show background in the plots
 
    for bin_index in bin_indices:
       
        bin_index_graph = bin_index+1
 
        first_bin_nominal_signal = histogram_nominal_signal[bin_index]
        first_bin_nominal_background = histogram_nominal_background[bin_index]
 
        delta_S_signal = []
        delta_S_background = []
 
        jes_range = np.linspace(0.9, 1.1, 21)
        for idx_jes in jes_range:
            # Signal
            syst_set = systematics(train_set, jes = idx_jes)
            target = syst_set["labels"]
            signal_field = syst_set["data"][target == 1]
            background_field = syst_set["data"][target == 0]
            score_signal = model.predict(signal_field)
            weights_signal = syst_set["weights"][target == 1]
            histogram_signal, _ = np.histogram(
                score_signal, bins=bins, range=(0, 1), weights=weights_signal
            )
 
            # Background
            score_background = model.predict(background_field)
            weights_background = syst_set["weights"][target == 0]
            histogram_background, _ = np.histogram(
                score_background, bins=bins, range=(0, 1), weights=weights_background
            )
 
            first_bin_signal = histogram_signal[bin_index]
            first_bin_background = histogram_background[bin_index]
            delta_signal = first_bin_signal - first_bin_nominal_signal
            delta_background = first_bin_background - first_bin_nominal_background
 
            delta_S_signal.append(delta_signal)
            delta_S_background.append(delta_background)
 
        plt.figure(figsize=(20, 10))
        plt.scatter(jes_range, delta_S_signal, label="Signal", color="blue")
        if show_background:
            plt.scatter(jes_range, delta_S_background, label='Background', color="orange")
        plt.xlabel("JES")
        plt.ylabel(r"$\Delta\ N$")
        plt.title(f"Shifted bin no. {bin_index_graph} of the Histogram")
 
        # Fit polynomial to delta_S_signal and delta_S_background
        fit_params_signal = fit_function(delta_S_signal)
        fit_params_background = fit_function(delta_S_background)
 
        # Generate smooth JES values for plotting the fit
        jes_smooth = np.linspace(0.9, 1.1, 100)
        fit_curve_signal = np.polyval(fit_params_signal, jes_smooth)
        fit_curve_background = np.polyval(fit_params_background, jes_smooth)
 
        # Plot the fit curves on top of the scatter plot
        plt.plot(jes_smooth, fit_curve_signal, label="Signal fit", color="blue", linestyle="--")
        if show_background:
            plt.plot(jes_smooth, fit_curve_background, label="Background fit", color="orange", linestyle="--")
        plt.legend()
        plt.grid()
        plt.title(f"Shifted bin no. {bin_index_graph} of the Histogram")
        os.makedirs("bin_graphs", exist_ok=True)
        if show_background:
            plt.savefig(f"bin_graphs/jes_analysis_bin_{bin_index_graph}_with_bg.png")
        else:
            plt.savefig(f"bin_graphs/jes_analysis_bin_{bin_index_graph}.png")
        plt.close()
 
 
    ######## Deux fonctions à regrouper
    ##### Il faudra appeler fit_function pour tous les bins à étudier et stocker les paramètres renvoyés dans une liste (le résultat est une liste de liste prise comme l'argument fitting_pol pour la 2de fonction)
 
    def eval_alpha(alpha, delta_S_signal, delta_S_background, maxi=3):
        """input :
        alpha -> the estimation of the parameter wanted
 
        fitting_pol -> list of the polynoms which fit the plots of Delta S as a function of the parameter alpha (Highest degree last) the signal in first, the background after
 
        output :
        list of S + Delta S for each bin"""
 
 
        fitting_pol = []
        for i in range(len(delta_S_signal)):
            fitting_pol.append(fit_function(delta_S_signal[i]), maxi=maxi)
 
        for i in range(len(delta_S_background)):
            fitting_pol.append(fit_function(delta_S_background[i]), maxi=maxi)
 
        list_S_plus_delta_S = []
        for i in range(len(fitting_pol)):
            list_S_plus_delta_S.append(np.polyval(fitting_pol[i][::-1], alpha))
        return list_S_plus_delta_S
 
    return fit_function, eval_alpha