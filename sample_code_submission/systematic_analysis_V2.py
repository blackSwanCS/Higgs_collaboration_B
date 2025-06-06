import numpy as np
from HiggsML.systematics import systematics
from model import Model
from HiggsML.datasets import download_dataset
from utils import histogram_dataset
import matplotlib.pyplot as plt
import seaborn as sns

data = download_dataset("blackSwan_data")

data.load_train_set()
data_set = data.get_train_set()


def tes_fitter(model, train_set, alpha_tes=1.03, nbin=25):
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

            syst_set = systematics(train_set, tes)
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
    # Write a function to loop over different values of tes and histogram and make fit function which transforms the histogram for any given TES

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

    # return (fit_function, delta_S_signal, delta_S_background)

    def give_fitting_functions(fitfunction, delta_S_signal, delta_S_background, maxi=2):
        """returns the coeeficients for the polynomial fitting function for all bins"""
        signal_fitting_pol = [
            fitfunction(delta_S_signal[i], maxi=maxi)
            for i in range(len(delta_S_signal))
        ]
        background_fitting_pol = [
            fitfunction(delta_S_background[i], maxi=maxi)
            for i in range(len(delta_S_background))
        ]
        return (signal_fitting_pol, background_fitting_pol)

    signal_fitting_pol, background_fitting_pol = give_fitting_functions(
        fit_function, delta_S_signal, delta_S_background, maxi=2
    )

    def shifted_score(alpha, signal_fitting_pol, background_fitting_pol):
        """returns the shifted values for the score"""
        syst_set = systematics(train_set, tes=alpha)
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

        shifted_histogram_signal = histogram_signal + delta_N_signal
        shifted_histogram_background = histogram_background + delta_N_background

        return (
            shifted_histogram_signal,
            shifted_histogram_background,
            histogram_signal,
            histogram_background,
        )

    (
        shifted_histogram_signal,
        shifted_histogram_background,
        histogram_signal,
        histogram_background,
    ) = shifted_score(alpha_tes, signal_fitting_pol, background_fitting_pol)

    # return(shifted_histogram_signal, shifted_histogram_background)

    def plot_shifted_score_histograms(
        shifted_signal,
        shifted_background,
        nominal_signal,
        nominal_background,
        nbin=25,
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
        nominal_signal = normalize(nominal_signal)
        nominal_background = normalize(nominal_background)

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
        axes[0].set_title("Nominal vs Shifted Histograms (normalized)")
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
        axes[1].set_title("Difference (Shifted - Nominal, normalized)")
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    plot_shifted_score_histograms(
        shifted_histogram_signal,
        shifted_histogram_background,
        histogram_signal,
        histogram_background,
        nbin=25,
        range_=(0, 1),
    )


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
    syst_set = systematics(train_set, jes=1)
    score = model.predict(syst_set["data"])

    histogram = np.histogram(score, bins=25, range=(0, 1))

    # Write a function to loop over different values of jes and histogram and make fit function which transforms the histogram for any given JES

    def fit_function(array, jes):
        # Dummy fit function, replace with actual fitting procedure
        return array * jes

    return fit_function
