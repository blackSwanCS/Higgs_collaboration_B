
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import joblib
import os
from pathlib import Path
import numpy as np

def amsasimov(s_in,b_in): 
    """
    asimov significance arXiv:1007.1727 eq. 97 (reduces to s/sqrt(b) if s<<b) 
    """
    # if b==0 ams is undefined, but return 0 without warning for convenience (hack)
    s=np.copy(s_in)
    b=np.copy(b_in)
    s=np.where( (b_in == 0) , 0., s_in)
    b=np.where( (b_in == 0) , 1., b)

    ams = np.sqrt(2*((s+b)*np.log(1+s/b)-s))
    ams=np.where( (s < 0)  | (b < 0), np.nan, ams) # nan if unphysical values.
    if np.isscalar(s_in):
        return float(ams)
    else:
        return  ams

def significance_vscore(y_true, y_score, sample_weight=None):
    """
    Calculate the significance using the Asimov method.
    """
    if sample_weight is None:
        # Provide a default value of 1.
        sample_weight = np.full(len(y_true), 1.)

    # Define bins for y_score, adapt the number as needed for your data
    bins = np.linspace(0, 1., 101)


    # Fills s and b weighted binned distributions
    s_hist, bin_edges = np.histogram(y_score[y_true == 1], bins=bins, weights=sample_weight[y_true == 1])
    b_hist, bin_edges = np.histogram(y_score[y_true == 0], bins=bins, weights=sample_weight[y_true == 0])


    # Compute cumulative sums (from the right!)
    s_cumul = np.cumsum(s_hist[::-1])[::-1]
    b_cumul = np.cumsum(b_hist[::-1])[::-1]

    # Compute significance
    significance=amsasimov(s_cumul,b_cumul)

    # Find the bin with the maximum significance
    max_value = np.max(significance)

    return significance
    

class BoostedDecisionTree:
    """
    This Dummy class implements a decision tree classifier
    change the code in the fit method to implement a decision tree classifier


    """

    def __init__(self, train_data):
        self.model = XGBClassifier()
        #self.model = XGBClassifier(learning_rate=0.36954584046859273,max_depth=6,n_estimators=194,use_label_encoder=False, eval_metric='logloss')
        self.scaler = StandardScaler()

    def save(self):
        """
        Save the model and scaler to the current directory.
        """
        # Obtenir le chemin absolu du dossier courant (du notebook)
        base_dir = Path().resolve()
        
        # Créer les répertoires si besoin
        models_dir = base_dir / 'models'
        scalers_dir = base_dir / 'scalers'
        models_dir.mkdir(exist_ok=True)
        scalers_dir.mkdir(exist_ok=True)

        # Sauvegarder les fichiers
        joblib.dump(self.model, models_dir / 'model.pkl')
        joblib.dump(self.scaler, scalers_dir / 'scaler.pkl')

        
    def fit(self, train_data, labels, weights=None):
        """
        Fit the model to the training data.
        """
        self.scaler.fit_transform(train_data)
        X_train_data = self.scaler.transform(train_data)
        self.model.fit(X_train_data, labels, weights)
        self.save()

    def predict(self, test_data):
        """ 
            Predict the labels for the test data.
            This method applies the same scaling to the test data as was applied to the training data.
        """
        test_data = self.scaler.transform(test_data)
        return self.model.predict_proba(test_data)[:, 1]
        
    def load(self, models_dir, scalers_dir):
        """
        Load the model from the specified path.
        """
        model_path = Path(models_dir) / 'model.pkl'
        scaler_path = Path(scalers_dir) / 'scaler.pkl'

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        
    def evaluate_AUC(self, test_data, labels):
        """
        Evaluate the model using AUC.
        """
        predictions = self.predict(test_data)
        return roc_auc_score(labels, predictions)
    
    def evaluate_significance(self, test_data, labels):
        """ 
        Evaluate the model using significance.
        """
        predictions = self.predict(test_data)
        return significance_vscore(labels, predictions)