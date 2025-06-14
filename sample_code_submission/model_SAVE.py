# ------------------------------
# Dummy Sample Submission
# ------------------------------

BDT = True
NN = False

from statistical_analysis import calculate_saved_info, compute_mu
import numpy as np


def amsasimov(s_in, b_in):
    """
    asimov significance arXiv:1007.1727 eq. 97 (reduces to s/sqrt(b) if s<<b)
    """
    # if b==0 ams is undefined, but return 0 without warning for convenience (hack)
    s = np.copy(s_in)
    b = np.copy(b_in)
    s = np.where((b_in == 0), 0.0, s_in)
    b = np.where((b_in == 0), 1.0, b)

    ams = np.sqrt(2 * ((s + b) * np.log(1 + s / b) - s))
    ams = np.where((s < 0) | (b < 0), np.nan, ams)  # nan if unphysical values.
    if np.isscalar(s_in):
        return float(ams)
    else:
        return ams


def significance_vscore(y_true, y_score, sample_weight=None):
    """
    Calculate the significance using the Asimov method.
    """
    if sample_weight is None:
        # Provide a default value of 1.
        sample_weight = np.full(len(y_true), 1.0)

    # Define bins for y_score, adapt the number as needed for your data
    bins = np.linspace(0, 1.0, 101)

    # Fills s and b weighted binned distributions
    s_hist, bin_edges = np.histogram(
        y_score[y_true == 1], bins=bins, weights=sample_weight[y_true == 1]
    )
    b_hist, bin_edges = np.histogram(
        y_score[y_true == 0], bins=bins, weights=sample_weight[y_true == 0]
    )

    # Compute cumulative sums (from the right!)
    s_cumul = np.cumsum(s_hist[::-1])[::-1]
    b_cumul = np.cumsum(b_hist[::-1])[::-1]

    # Compute significance
    significance = amsasimov(s_cumul, b_cumul)

    # Find the bin with the maximum significance
    max_value = np.max(significance)

    return significance


class Model:
    """
    This is a model class to be submitted by the participants in their submission.

    This class should consists of the following functions
    1) init :
        takes 3 arguments: train_set systematics and model_type.
        can be used for initializing variables, classifier etc.
    2) fit :
        takes no arguments
        can be used to train a classifier
    3) predict:
        takes 1 argument: test sets
        can be used to get predictions of the test set.
        returns a dictionary

    Note:   Add more methods if needed e.g. save model, load pre-trained model etc.
            It is the participant's responsibility to make sure that the submission
            class is named "Model" and that its constructor arguments remains the same.
            The ingestion program initializes the Model class and calls fit and predict methods

            When you add another file with the submission model e.g. a trained model to be loaded and used,
            load it in the following way:

            # get to the model directory (your submission directory)
            model_dir = os.path.dirname(os.path.abspath(__file__))

            your trained model file is now in model_dir, you can load it from here
    """

    def __init__(self, get_train_set=None, systematics=None, model_type="sample_model"):
        """
        Model class constructor

        Params:
            train_set:
                a dictionary with data, labels, weights and settings

            systematics:
                a class which you can use to get a dataset with systematics added
                See sample submission for usage of systematics


        Returns:
            None
        """

        indices = np.arange(15000)

        np.random.shuffle(indices)

        train_indices = indices[:5000]
        holdout_indices = indices[5000:10000]
        valid_indices = indices[10000:]

        training_df = get_train_set(selected_indices=train_indices)

        self.training_set = {
            "labels": training_df.pop("labels"),
            "weights": training_df.pop("weights"),
            "detailed_labels": training_df.pop("detailed_labels"),
            "data": training_df,
        }

        del training_df

        self.systematics = systematics

        print("Training Data: ", self.training_set["data"].shape)
        print("Training Labels: ", self.training_set["labels"].shape)
        print("Training Weights: ", self.training_set["weights"].shape)
        print(
            "sum_signal_weights: ",
            self.training_set["weights"][self.training_set["labels"] == 1].sum(),
        )
        print(
            "sum_bkg_weights: ",
            self.training_set["weights"][self.training_set["labels"] == 0].sum(),
        )

        valid_df = get_train_set(selected_indices=valid_indices)

        self.valid_set = {
            "labels": valid_df.pop("labels"),
            "weights": valid_df.pop("weights"),
            "detailed_labels": valid_df.pop("detailed_labels"),
            "data": valid_df,
        }
        del valid_df

        print()
        print("Valid Data: ", self.valid_set["data"].shape)
        print("Valid Labels: ", self.valid_set["labels"].shape)
        print("Valid Weights: ", self.valid_set["weights"].shape)
        print(
            "sum_signal_weights: ",
            self.valid_set["weights"][self.valid_set["labels"] == 1].sum(),
        )
        print(
            "sum_bkg_weights: ",
            self.valid_set["weights"][self.valid_set["labels"] == 0].sum(),
        )

        holdout_df = get_train_set(selected_indices=holdout_indices)

        self.holdout_set = {
            "labels": holdout_df.pop("labels"),
            "weights": holdout_df.pop("weights"),
            "detailed_labels": holdout_df.pop("detailed_labels"),
            "data": holdout_df,
        }

        del holdout_df

        print()
        print("Holdout Data: ", self.holdout_set["data"].shape)
        print("Holdout Labels: ", self.holdout_set["labels"].shape)
        print("Holdout Weights: ", self.holdout_set["weights"].shape)
        print(
            "sum_signal_weights: ",
            self.holdout_set["weights"][self.holdout_set["labels"] == 1].sum(),
        )
        print(
            "sum_bkg_weights: ",
            self.holdout_set["weights"][self.holdout_set["labels"] == 0].sum(),
        )
        print(" \n ")

        print("Training Data: ", self.training_set["data"].shape)
        print(f"DEBUG: model_type = {repr(model_type)}")

        if model_type == "BDT":
            from boosted_decision_tree import BoostedDecisionTree

            self.model = BoostedDecisionTree(train_data=self.training_set["data"])
        elif model_type == "NN":
            from neural_network import NeuralNetwork

            self.model = NeuralNetwork(train_data=self.training_set["data"])
        elif model_type == "sample_model":
            from sample_model import SampleModel

            self.model = SampleModel()
        else:
            print(f"model_type {model_type} not found")
            raise ValueError(f"model_type {model_type} not found")
        self.name = model_type

        print(f" Model is { self.name}")

    def fit(self):
        """
        Params:
            None

        Functionality:
            this function can be used to train a model

        Returns:
            None
        """

        balanced_set = self.training_set.copy()

        weights_train = self.training_set["weights"].copy()
        train_labels = self.training_set["labels"].copy()
        class_weights_train = (
            weights_train[train_labels == 0].sum(),
            weights_train[train_labels == 1].sum(),
        )

        for i in range(len(class_weights_train)):  # loop on B then S target
            # training dataset: equalize number of background and signal
            weights_train[train_labels == i] *= (
                max(class_weights_train) / class_weights_train[i]
            )
            # test dataset : increase test weight to compensate for sampling

        balanced_set["weights"] = weights_train

        self.model.fit(
            balanced_set["data"], balanced_set["labels"], balanced_set["weights"]
        )

        self.holdout_set = self.systematics(self.holdout_set)

        self.saved_info = calculate_saved_info(self.model, self.holdout_set)

        self.training_set = self.systematics(self.training_set)

        # Compute  Results
        train_score = self.model.predict(self.training_set["data"])
        train_results = compute_mu(
            train_score, self.training_set["weights"], self.saved_info
        )

        holdout_score = self.model.predict(self.holdout_set["data"])
        holdout_results = compute_mu(
            holdout_score, self.holdout_set["weights"], self.saved_info
        )

        self.valid_set = self.systematics(self.valid_set)

        valid_score = self.model.predict(self.valid_set["data"])

        valid_results = compute_mu(
            valid_score, self.valid_set["weights"], self.saved_info
        )

        print("Train Results: ")
        for key in train_results.keys():
            print("\t", key, " : ", train_results[key])

        print("Holdout Results: ")
        for key in holdout_results.keys():
            print("\t", key, " : ", holdout_results[key])

        print("Valid Results: ")
        for key in valid_results.keys():
            print("\t", key, " : ", valid_results[key])

        print("Significance (Asimov):")
        significance = significance_vscore(
            y_true=self.valid_set["labels"],
            y_score=valid_score,
            sample_weight=self.valid_set["weights"],
        )
        max_significance = np.nanmax(significance)
        print(f"\tMaximum Asimov significance: {max_significance:.4f}")

        self.valid_set["data"]["score"] = valid_score
        from utils import roc_curve_wrapper, histogram_dataset

        histogram_dataset(
            self.valid_set["data"],
            self.valid_set["labels"],
            self.valid_set["weights"],
            columns=["score"],
        )

        from HiggsML.visualization import stacked_histogram

        stacked_histogram(
            self.valid_set["data"],
            self.valid_set["labels"],
            self.valid_set["weights"],
            self.valid_set["detailed_labels"],
            "score",
        )

        roc_curve_wrapper(
            score=valid_score,
            labels=self.valid_set["labels"],
            weights=self.valid_set["weights"],
            plot_label="valid_set" + self.name,
        )

    def predict(self, test_set):
        """
        Params:
            test_set

        Functionality:
            this function can be used for predictions using the test sets

        Returns:
            dict with keys
                - mu_hat
                - delta_mu_hat
                - p16
                - p84
        """

        test_data = test_set["data"]
        test_weights = test_set["weights"]

        predictions = self.model.predict(test_data)

        result_mu_cal = compute_mu(predictions, test_weights, self.saved_info)

        print("Test Results: ", result_mu_cal)

        result = {
            "mu_hat": result_mu_cal["mu_hat"],
            "delta_mu_hat": result_mu_cal["del_mu_tot"],
            "p16": result_mu_cal["mu_hat"] - result_mu_cal["del_mu_tot"],
            "p84": result_mu_cal["mu_hat"] + result_mu_cal["del_mu_tot"],
        }

        return result


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import keras_tuner as kt


class NeuralNetworkTunable:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.input_dim = None

    def build_model(self, hp):
        model = Sequential()

        # Nombre de couches cachées
        num_layers = hp.Int("num_layers", 1, 4)

        # Première couche avec input_shape
        model.add(
            Dense(
                units=hp.Int("units_0", 16, 128, step=16),
                activation=hp.Choice("activation_0", ["relu", "tanh"]),
                input_shape=(self.input_dim,),
            )
        )

        # Couches cachées supplémentaires
        for i in range(1, num_layers):
            model.add(
                Dense(
                    units=hp.Int(f"units_{i}", 16, 128, step=16),
                    activation=hp.Choice(f"activation_{i}", ["relu", "tanh"]),
                )
            )

        # Couche de sortie
        model.add(Dense(1, activation="sigmoid"))

        # Taux d'apprentissage
        lr = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def fit(self, X_train, y_train, weights_train=None):
        self.input_dim = X_train.shape[1]

        # Normalisation
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)

        # HPO
        tuner = kt.RandomSearch(
            self.build_model,
            objective="val_accuracy",
            max_trials=10,
            executions_per_trial=1,
            directory="hpo_dir",
            project_name="deep_nn_tuning",
        )

        # Batch size comme hyperparamètre
        batch_size = 64  # Valeur par défaut
        try:
            batch_size = tuner.oracle.hyperparameters.Int(
                "batch_size", 32, 128, step=32
            )
        except:
            pass  # pas indispensable si on ne le tune pas

        tuner.search(
            X_train_scaled,
            y_train,
            sample_weight=weights_train,
            epochs=15,
            validation_split=0.2,
            batch_size=batch_size,
            verbose=1,
        )

        self.model = tuner.get_best_models(1)[0]

    def predict(self, X_test):
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled).flatten()


import optuna
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from HiggsML.datasets import download_dataset

# Chargement des données
get_train_set = download_dataset("blackSwan_data")

indices = np.arange(15000)
np.random.shuffle(indices)
train_idx, valid_idx = indices[:5000], indices[5000:6000]

train_df = get_train_set
valid_df = get_train_set

X_train = train_df.drop(columns=["labels", "weights", "detailed_labels"])
y_train = train_df["labels"]
w_train = train_df["weights"]

X_valid = valid_df.drop(columns=["labels", "weights", "detailed_labels"])
y_valid = valid_df["labels"]
w_valid = valid_df["weights"]

# Standardisation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)

model = NeuralNetworkTunable()
model.fit(X_train, y_train, weights_train)
y_pred = model.predict(X_test)


# Function objectif pour Optuna
def objective(trial):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    hidden_units = trial.suggest_int("hidden_units", 8, 128)
    activation = trial.suggest_categorical("activation", ["relu", "tanh"])
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    epochs = trial.suggest_int("epochs", 5, 20)

    model = Sequential()
    model.add(Dense(hidden_units, activation=activation, input_dim=X_train.shape[1]))
    for _ in range(n_layers - 1):
        model.add(Dense(hidden_units, activation=activation))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=BinaryCrossentropy(),
        metrics=["accuracy"],
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=w_train,
        epochs=epochs,
        batch_size=128,
        verbose=0,
    )

    preds = model.predict(X_valid).ravel()
    score = roc_auc_score(y_valid, preds, sample_weight=w_valid)
    return score


# Lancement de l'optimisation
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

# Affichage des meilleurs résultats
print("Best trial:")
best = study.best_trial
for key, value in best.params.items():
    print(f"{key}: {value}")
print(f"Best AUC: {best.value:.4f}")
