import numpy as np
from HiggsML.systematics import systematics
import matplotlib.pyplot as plt


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
    
    syst_set = systematics(train_set, 1)
    
    target = syst_set["labels"]
    signal_field = syst_set["data"][target == 1]
    background_field = syst_set["data"][target == 0]
    
  
    score_signal = model.predict(signal_field)
    signal_weights = syst_set["weights"][target==1]
    histogram_nominal_signal, _ = np.histogram(score_signal, bins=100, range=(0, 1), weights=signal_weights)

    score_background = model.predict(background_field)
    background_weights = syst_set["weights"][target==0]
    histogram_nominal_background, _ = np.histogram(score_background, bins=100, range=(0, 1), weights=background_weights)


    first_bin_nominal_signal = histogram_nominal_signal[0]
    first_bin_nominal_background = histogram_nominal_background[0]


    delta_S_signal = []
    delta_S_background = []
    
    tes_range = np.linspace(0.9, 1.1, 101)
    for tes in tes_range:
        # Signal
        syst_set = systematics(train_set, tes)
        target = syst_set["labels"]
        signal_field = syst_set["data"][target == 1]
        background_field = syst_set["data"][target == 0]
        score_signal = model.predict(signal_field)
        if isinstance(signal_field, dict):
            weights_signal = syst_set["weights"][target==1]
        else:
            weights_signal = syst_set["weights"][target==1] if "weights" in signal_field.columns else np.ones(len(score_signal))
        histogram_signal, _ = np.histogram(score_signal, bins=100, range=(0, 1), weights=weights_signal)

        # Background
        score_background = model.predict(background_field)
        if isinstance(background_field, dict):
            weights_background = syst_set["weights"][target==0]
        else:
            weights_background = syst_set["weights"][target==0] if "weights" in background_field.columns else np.ones(len(score_background))
        histogram_background, _ = np.histogram(score_background, bins=100, range=(0, 1), weights=weights_background)
        
        first_bin_signal = histogram_signal[0]
        first_bin_background = histogram_background[0]
        
        # print(histogram_signal)
        print("first_bin_signal:", first_bin_signal)
        print("first_bin_nominal_signal : ", first_bin_nominal_signal)
        
        delta_signal = first_bin_signal - first_bin_nominal_signal
        delta_background = first_bin_background - first_bin_nominal_background
        
        delta_S_signal.append(delta_signal)
        delta_S_background.append(delta_background)
    
    plt.figure(figsize=(10, 5))
    plt.plot(tes_range, delta_S_signal, label='Signal')
    #plt.plot(tes_range, delta_S_background, label='Background')
    plt.xlabel('TES')
    plt.ylabel('Delta S')
    plt.title('TES Uncertainty Analysis')
    plt.legend()
    plt.grid()
    plt.show()

    


    # Write a function to loop over different values of tes and histogram and make fit function which transforms the histogram for any given TES

    def fit_function(array, tes):
        # Dummy fit function, replace with actual fitting procedure
        return array * tes

    return fit_function


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

    histogram = np.histogram(score, bins=100, range=(0, 1))

    # Write a function to loop over different values of jes and histogram and make fit function which transforms the histogram for any given JES

    def fit_function(array, jes):
        # Dummy fit function, replace with actual fitting procedure
        return array * jes

    return fit_function

