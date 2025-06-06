import numpy as np
from HiggsML.systematics import systematics
import matplotlib.pyplot as plt
import seaborn as sns
import os


def fit_function(array, range_, maxi=2):
    """tes et jes sont toujours défini entre 0.9 et 1.1
    array correspond à la liste des valeurs prise par l'histogramme après isolement d'un bin
    """
    meilleur = 1
    R_meilleur = np.inf
    R = 0
    for deg in range(0, maxi + 1):
        parameters = np.polyfit(range_, array, deg)
        # print(f"{deg} : {parameters}")
        for ind in range(len(range_)):
            y = 0
            for i in range(deg + 1):
                y += parameters[i] * range_[ind] ** (deg - i)
            R += (y - array[ind]) ** 2
        if R < R_meilleur:
            meilleur = deg
            R_meilleur = R
        R = 0
    # print("meilleur :", meilleur)

    return np.polyfit(range_, array, meilleur)


def tes_fitter(model, train_set, nbin=10, get_plots=True):
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

    nominal_syst_set = systematics(train_set, tes=1)

    target = nominal_syst_set["labels"]
    signal_field = nominal_syst_set["data"][target == 1]
    background_field = nominal_syst_set["data"][target == 0]

    score_signal = model.predict(signal_field)
    signal_weights = nominal_syst_set["weights"][target == 1]
    histogram_nominal_signal, _ = np.histogram(
        score_signal, bins=nbin, range=(0, 1), weights=signal_weights
    )

    score_background = model.predict(background_field)
    background_weights = nominal_syst_set["weights"][target == 0]
    histogram_nominal_background, _ = np.histogram(
        score_background, bins=nbin, range=(0, 1), weights=background_weights
    )

    delta_S_signal = []
    delta_S_background = []

    tes_range = np.linspace(0.9, 1.1, 10)

    for i in range(nbin):
        bin_delta_S_signal = []
        bin_delta_S_background = []
        for tes in tes_range:

            syst_set = systematics(train_set, tes=tes)
            target = syst_set["labels"]
            signal_field = syst_set["data"][target == 1]
            background_field = syst_set["data"][target == 0]

            # Signal
            score_signal = model.predict(signal_field)
            weights_signal = syst_set["weights"][target == 1]
            histogram_signal, _ = np.histogram(
                score_signal, bins=nbin, range=(0, 1), weights=weights_signal
            )

            # Background
            score_background = model.predict(background_field)
            weights_background = syst_set["weights"][target == 0]
            histogram_background, _ = np.histogram(
                score_background, bins=nbin, range=(0, 1), weights=weights_background
            )

            bin_signal = histogram_signal[i]
            bin_background = histogram_background[i]
            bin_nominal_signal = histogram_nominal_signal[i]
            bin_nominal_background = histogram_nominal_background[i]
            delta_signal = bin_signal - bin_nominal_signal
            delta_background = bin_background - bin_nominal_background

            bin_delta_S_signal.append(delta_signal)
            bin_delta_S_background.append(delta_background)
        delta_S_signal.append(bin_delta_S_signal)
        delta_S_background.append(bin_delta_S_background)

        if get_plots and any(
            (i == 0, i == nbin // 2 - 1, i == nbin - 1)
        ):  # Plot only for the first, middle, and last bins
            # Plot Signal
            plt.figure(figsize=(10, 5))
            plt.scatter(
                tes_range, delta_S_signal[i], label="Signal", color="blue", zorder=2
            )
            plt.xlabel("TES")
            plt.ylabel(r"$\Delta\ N$")
            plt.title(f"Shifted bin no. {i+1} of the Histogram (Signal)")

            # Fit polynomial to delta_S_signal
            fit_params_signal = fit_function(delta_S_signal[i], tes_range)
            tes_smooth = np.linspace(0.9, 1.1, 100)
            fit_curve_signal = np.polyval(fit_params_signal, tes_smooth)
            chi_squared_signal = np.sum(
                (np.polyval(fit_params_signal, tes_range) - delta_S_signal[i]) ** 2
            )

            plt.plot(
                tes_smooth,
                fit_curve_signal,
                label=f"Fit with $\\chi^2 = {chi_squared_signal:.3g}$",
                color="blue",
                linestyle="--",
                zorder=2,
            )
            plt.legend()
            plt.grid(True, zorder=1)
            os.makedirs("bin_graphs", exist_ok=True)
            plt.savefig(f"bin_graphs/tes_analysis_bin_{i+1}_sg.png")
            plt.close()

            # Plot Background
            plt.figure(figsize=(10, 5))
            plt.scatter(
                tes_range,
                delta_S_background[i],
                label="Background",
                color="red",
                zorder=2,
            )
            plt.xlabel("TES")
            plt.ylabel(r"$\Delta\ N$")
            plt.title(f"Shifted bin no. {i+1} of the Histogram (Background)")

            # Fit polynomial to delta_S_background
            fit_params_background = fit_function(delta_S_background[i], tes_range)
            fit_curve_background = np.polyval(fit_params_background, tes_smooth)
            chi_squared_background = np.sum(
                (np.polyval(fit_params_background, tes_range) - delta_S_background[i])
                ** 2
            )

            plt.plot(
                tes_smooth,
                fit_curve_background,
                label=f"Fit with $\\chi^2 = {chi_squared_background:.3g}$",
                color="red",
                linestyle="--",
                zorder=2,
            )
            plt.legend()
            plt.grid(True, zorder=1)
            plt.savefig(f"bin_graphs/tes_analysis_bin_{i+1}_bg.png")
            plt.close()

    # Write a function to loop over different values of tes and histogram and make fit function which transforms the histogram for any given TES

    # return (fit_function, delta_S_signal, delta_S_background)

    def give_fitting_functions(fitfunction, delta_S_signal, delta_S_background, maxi=2):
        """returns the coeeficients for the polynomial fitting function for all bins"""
        signal_fitting_pol = [
            fitfunction(delta_S_signal[i], tes_range, maxi=maxi)
            for i in range(len(delta_S_signal))
        ]
        background_fitting_pol = [
            fitfunction(delta_S_background[i], tes_range, maxi=maxi)
            for i in range(len(delta_S_background))
        ]
        return (signal_fitting_pol, background_fitting_pol)

    signal_fitting_pol, background_fitting_pol = give_fitting_functions(
        fit_function, delta_S_signal, delta_S_background, maxi=2
    )

    def shifted_score(alpha, signal_fitting_pol, background_fitting_pol):
        """returns the shifted values for the score"""

        delta_N_signal = np.array(
            [
                np.polyval(signal_fitting_pol[i][::-1], alpha)
                for i in range(len(signal_fitting_pol))
            ]
        )
        delta_N_background = np.array(
            [
                np.polyval(background_fitting_pol[i][::-1], alpha)
                for i in range(len(background_fitting_pol))
            ]
        )

        shifted_histogram_signal = histogram_nominal_signal + delta_N_signal
        shifted_histogram_background = histogram_nominal_background + delta_N_background

        return (shifted_histogram_signal, shifted_histogram_background)

    shifted_histogram_signal, shifted_histogram_background = shifted_score(
        1.03, signal_fitting_pol, background_fitting_pol
    )

    def shifted_score_signal_tes(alpha, signal_fitting_pol):
        """returns the shifted values for the score"""

        delta_N_signal = np.array(
            [
                np.polyval(signal_fitting_pol[i][::-1], alpha)
                for i in range(len(signal_fitting_pol))
            ]
        )

        shifted_histogram_signal = histogram_nominal_signal + delta_N_signal

        return shifted_histogram_signal

    def shifted_score_background_tes(alpha, background_fitting_pol):
        """returns the shifted values for the score"""

        delta_N_background = np.array(
            [
                np.polyval(background_fitting_pol[i][::-1], alpha)
                for i in range(len(background_fitting_pol))
            ]
        )

        shifted_histogram_background = histogram_nominal_background + delta_N_background

        return shifted_histogram_background

    # return(shifted_histogram_signal, shifted_histogram_background)

    def plot_shifted_score_histograms(
        shifted_signal,
        shifted_background,
        histogram_nominal_signal,
        histogram_nominal_background,
        nbin,
        range_=(0, 1),
    ):
        """
        Plot normalized (density) shifted and nominal histograms and their bin-wise differences.

        Args:
            shifted_signal, shifted_background: shifted histograms
            nominal_signal, nominal_background: nominal histograms
            nbin: number of bins
            range_: range of score values
        """
        sns.set_theme(style="whitegrid")
        bin_edges = np.linspace(range_[0], range_[1], nbin + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_width = bin_edges[1] - bin_edges[0]

        # Normalize all histograms to unit area
        def normalize(hist):
            area = np.sum(hist) * bin_width
            return hist / area if area > 0 else hist

        shifted_signal = normalize(shifted_signal)
        shifted_background = normalize(shifted_background)
        nominal_signal = normalize(histogram_nominal_signal)
        nominal_background = normalize(histogram_nominal_background)

        # Differences
        delta_signal = shifted_signal - nominal_signal
        delta_background = shifted_background - nominal_background

        fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

        # Top plot: normalized histograms
        axes[0].bar(
            bin_centers,
            nominal_signal,
            width=bin_width,
            alpha=0.2,
            color="blue",
            label="Nominal Signal",
        )
        axes[0].bar(
            bin_centers,
            shifted_signal,
            width=bin_width,
            alpha=0.6,
            color="blue",
            label="Shifted Signal",
            edgecolor="black",
        )

        axes[0].bar(
            bin_centers,
            nominal_background,
            width=bin_width,
            alpha=0.2,
            color="red",
            label="Nominal Background",
        )
        axes[0].bar(
            bin_centers,
            shifted_background,
            width=bin_width,
            alpha=0.6,
            color="red",
            label="Shifted Background",
            edgecolor="black",
        )

        axes[0].set_ylabel("Density")
        axes[0].set_title("Nominal vs Shifted Histograms (normalized) (tes)")
        axes[0].legend()

        # Bottom plot: differences (also in density units)
        axes[1].bar(
            bin_centers,
            delta_signal,
            width=bin_width,
            alpha=0.4,
            color="blue",
            label="Δ Signal",
        )
        axes[1].bar(
            bin_centers,
            delta_background,
            width=bin_width,
            alpha=0.4,
            color="red",
            label="Δ Background",
        )

        axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
        axes[1].set_xlabel("Score")
        axes[1].set_ylabel("Δ Density")
        axes[1].set_title("Difference (Shifted - Nominal, normalized) (tes)")
        axes[1].legend()

        os.makedirs("bin_graphs", exist_ok=True)
        plt.savefig("bin_graphs/shifted_histograms_tes.png")
        plt.tight_layout()
        plt.show()

    if get_plots:
        plot_shifted_score_histograms(
            shifted_histogram_signal,
            shifted_histogram_background,
            histogram_nominal_signal,
            histogram_nominal_background,
            nbin,
            range_=(0, 1),
        )

    return (
        shifted_score_signal_tes,
        shifted_score_background_tes,
        signal_fitting_pol,
        background_fitting_pol,
    )


def jes_fitter(model, train_set, nbin=10, get_plots=True):
    """
    Task 1 : Analysis JES Uncertainty
    1. Loop over different values of jes and make store the score
    2. Make a histogram of the score

    Task 2 : Fit the histogram
    1. Write a function to loop over different values of jes and histogram and make fit function for each bin in the histogram
    2. store the fit functions in an array
    3. return the fit functions

      histogram and make fit function which transforms the histogram for any given JES

    """

    nominal_syst_set = systematics(train_set, jes=1)

    target = nominal_syst_set["labels"]
    signal_field = nominal_syst_set["data"][target == 1]
    background_field = nominal_syst_set["data"][target == 0]

    score_signal = model.predict(signal_field)
    signal_weights = nominal_syst_set["weights"][target == 1]
    histogram_nominal_signal, _ = np.histogram(
        score_signal, bins=nbin, range=(0, 1), weights=signal_weights
    )

    score_background = model.predict(background_field)
    background_weights = nominal_syst_set["weights"][target == 0]
    histogram_nominal_background, _ = np.histogram(
        score_background, bins=nbin, range=(0, 1), weights=background_weights
    )

    delta_S_signal = []
    delta_S_background = []

    jes_range = np.linspace(0.9, 1.1, 10)

    for i in range(nbin):
        bin_delta_S_signal = []
        bin_delta_S_background = []
        for jes in jes_range:

            syst_set = systematics(train_set, jes=jes)
            target = syst_set["labels"]
            signal_field = syst_set["data"][target == 1]
            background_field = syst_set["data"][target == 0]

            # Signal
            score_signal = model.predict(signal_field)
            weights_signal = syst_set["weights"][target == 1]
            histogram_signal, _ = np.histogram(
                score_signal, bins=nbin, range=(0, 1), weights=weights_signal
            )

            # Background
            score_background = model.predict(background_field)
            weights_background = syst_set["weights"][target == 0]
            histogram_background, _ = np.histogram(
                score_background, bins=nbin, range=(0, 1), weights=weights_background
            )

            bin_signal = histogram_signal[i]
            bin_background = histogram_background[i]
            bin_nominal_signal = histogram_nominal_signal[i]
            bin_nominal_background = histogram_nominal_background[i]
            delta_signal = bin_signal - bin_nominal_signal
            delta_background = bin_background - bin_nominal_background

            bin_delta_S_signal.append(delta_signal)
            bin_delta_S_background.append(delta_background)
        delta_S_signal.append(bin_delta_S_signal)
        delta_S_background.append(bin_delta_S_background)

        if get_plots and any(
            (i == 0, i == nbin // 2 - 1, i == nbin - 1)
        ):  # Plot only for the first, middle, and last bins
            # Plot Signal
            plt.figure(figsize=(10, 5))
            plt.scatter(
                jes_range, delta_S_signal[i], label="Signal", color="blue", zorder=2
            )
            plt.xlabel("JES")
            plt.ylabel(r"$\Delta\ N$")
            plt.title(f"Shifted bin no. {i+1} of the Histogram (Signal)")

            # Fit polynomial to delta_S_signal
            fit_params_signal = fit_function(delta_S_signal[i], jes_range)
            jes_smooth = np.linspace(0.9, 1.1, 100)
            fit_curve_signal = np.polyval(fit_params_signal, jes_smooth)
            chi_squared_signal = np.sum(
                (np.polyval(fit_params_signal, jes_range) - delta_S_signal[i]) ** 2
            )

            plt.plot(
                jes_smooth,
                fit_curve_signal,
                label=f"Fit with $\\chi^2 = {chi_squared_signal:.3g}$",
                color="blue",
                linestyle="--",
                zorder=2,
            )
            plt.legend()
            plt.grid(True, zorder=1)
            os.makedirs("bin_graphs", exist_ok=True)
            plt.savefig(f"bin_graphs/jes_analysis_bin_{i+1}_sg.png")
            plt.close()

            # Plot Background
            plt.figure(figsize=(10, 5))
            plt.scatter(
                jes_range,
                delta_S_background[i],
                label="Background",
                color="red",
                zorder=2,
            )
            plt.xlabel("JES")
            plt.ylabel(r"$\Delta\ N$")
            plt.title(f"Shifted bin no. {i+1} of the Histogram (Background)")

            # Fit polynomial to delta_S_background
            fit_params_background = fit_function(delta_S_background[i], jes_range)
            fit_curve_background = np.polyval(fit_params_background, jes_smooth)
            chi_squared_background = np.sum(
                (np.polyval(fit_params_background, jes_range) - delta_S_background[i])
                ** 2
            )

            plt.plot(
                jes_smooth,
                fit_curve_background,
                label=f"Fit with $\\chi^2 = {chi_squared_background:.3g}$",
                color="red",
                linestyle="--",
                zorder=2,
            )
            plt.legend()
            plt.grid(True, zorder=1)
            plt.savefig(f"bin_graphs/jes_analysis_bin_{i+1}_bg.png")
            plt.close()

    # Write a function to loop over different values of jes and histogram and make fit function which transforms the histogram for any given JES

    # return (fit_function, delta_S_signal, delta_S_background)

    def give_fitting_functions(fitfunction, delta_S_signal, delta_S_background, maxi=2):
        """returns the coeeficients for the polynomial fitting function for all bins"""
        signal_fitting_pol = [
            fitfunction(delta_S_signal[i], jes_range, maxi=maxi)
            for i in range(len(delta_S_signal))
        ]
        background_fitting_pol = [
            fitfunction(delta_S_background[i], jes_range, maxi=maxi)
            for i in range(len(delta_S_background))
        ]
        return (signal_fitting_pol, background_fitting_pol)

    signal_fitting_pol, background_fitting_pol = give_fitting_functions(
        fit_function, delta_S_signal, delta_S_background, maxi=2
    )

    def shifted_score(alpha, signal_fitting_pol, background_fitting_pol):
        """returns the shifted values for the score"""

        delta_N_signal = np.array(
            [
                np.polyval(signal_fitting_pol[i][::-1], alpha)
                for i in range(len(signal_fitting_pol))
            ]
        )
        delta_N_background = np.array(
            [
                np.polyval(background_fitting_pol[i][::-1], alpha)
                for i in range(len(background_fitting_pol))
            ]
        )

        shifted_histogram_signal = histogram_nominal_signal + delta_N_signal
        shifted_histogram_background = histogram_nominal_background + delta_N_background

        return (shifted_histogram_signal, shifted_histogram_background)

    shifted_histogram_signal, shifted_histogram_background = shifted_score(
        1.03, signal_fitting_pol, background_fitting_pol
    )

    def shifted_score_signal_jes(alpha, signal_fitting_pol):
        """returns the shifted values for the score"""

        delta_N_signal = np.array(
            [
                np.polyval(signal_fitting_pol[i][::-1], alpha)
                for i in range(len(signal_fitting_pol))
            ]
        )

        shifted_histogram_signal = histogram_nominal_signal + delta_N_signal

        return shifted_histogram_signal

    def shifted_score_background_jes(alpha, background_fitting_pol):
        """returns the shifted values for the score"""

        delta_N_background = np.array(
            [
                np.polyval(background_fitting_pol[i][::-1], alpha)
                for i in range(len(background_fitting_pol))
            ]
        )

        shifted_histogram_background = histogram_nominal_background + delta_N_background

        return shifted_histogram_background

    # return(shifted_histogram_signal, shifted_histogram_background)

    def plot_shifted_score_histograms(
        shifted_signal,
        shifted_background,
        histogram_nominal_signal,
        histogram_nominal_background,
        nbin,
        range_=(0, 1),
    ):
        """
        Plot normalized (density) shifted and nominal histograms and their bin-wise differences.

        Args:
            shifted_signal, shifted_background: shifted histograms
            nominal_signal, nominal_background: nominal histograms
            nbin: number of bins
            range_: range of score values
        """
        sns.set_theme(style="whitegrid")
        bin_edges = np.linspace(range_[0], range_[1], nbin + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_width = bin_edges[1] - bin_edges[0]

        # Normalize all histograms to unit area
        def normalize(hist):
            area = np.sum(hist) * bin_width
            return hist / area if area > 0 else hist

        shifted_signal = normalize(shifted_signal)
        shifted_background = normalize(shifted_background)
        nominal_signal = normalize(histogram_nominal_signal)
        nominal_background = normalize(histogram_nominal_background)

        # Differences
        delta_signal = shifted_signal - nominal_signal
        delta_background = shifted_background - nominal_background

        fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

        # Top plot: normalized histograms
        axes[0].bar(
            bin_centers,
            nominal_signal,
            width=bin_width,
            alpha=0.2,
            color="blue",
            label="Nominal Signal",
        )
        axes[0].bar(
            bin_centers,
            shifted_signal,
            width=bin_width,
            alpha=0.6,
            color="blue",
            label="Shifted Signal",
            edgecolor="black",
        )

        axes[0].bar(
            bin_centers,
            nominal_background,
            width=bin_width,
            alpha=0.2,
            color="red",
            label="Nominal Background",
        )
        axes[0].bar(
            bin_centers,
            shifted_background,
            width=bin_width,
            alpha=0.6,
            color="red",
            label="Shifted Background",
            edgecolor="black",
        )

        axes[0].set_ylabel("Density")
        axes[0].set_title("Nominal vs Shifted Histograms (normalized) (jes)")
        axes[0].legend()

        # Bottom plot: differences (also in density units)
        axes[1].bar(
            bin_centers,
            delta_signal,
            width=bin_width,
            alpha=0.4,
            color="blue",
            label="Δ Signal",
        )
        axes[1].bar(
            bin_centers,
            delta_background,
            width=bin_width,
            alpha=0.4,
            color="red",
            label="Δ Background",
        )

        axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
        axes[1].set_xlabel("Score")
        axes[1].set_ylabel("Δ Density")
        axes[1].set_title("Difference (Shifted - Nominal, normalized) (jes)")
        axes[1].legend()

        os.makedirs("bin_graphs", exist_ok=True)
        plt.savefig("bin_graphs/shifted_histograms_jes.png")
        plt.tight_layout()
        plt.show()

    if get_plots:
        plot_shifted_score_histograms(
            shifted_histogram_signal,
            shifted_histogram_background,
            histogram_nominal_signal,
            histogram_nominal_background,
            nbin,
            range_=(0, 1),
        )

    return (
        shifted_score_signal_jes,
        shifted_score_background_jes,
        signal_fitting_pol,
        background_fitting_pol,
    )
