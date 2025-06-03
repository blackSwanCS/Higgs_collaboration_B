from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib


class NeuralNetwork:
    """
    This Dummy class implements a neural network classifier
    change the code in the fit method to implement a neural network classifier


    """

    def __init__(self, train_data, model_path=None):
        self.model = Sequential()

        n_dim = train_data.shape[1]

        self.model.add(Dense(10, input_dim=n_dim, activation="relu"))
        self.model.add(Dense(10, activation="relu"))
        self.model.add(Dense(1, activation="sigmoid"))

        self.model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        
        if model_path is None:
            self.model_path = os.path.join(os.path.dirname(__file__), "model.h5")
        else:
            self.model_path = model_path
        self.scaler = StandardScaler()


    def save_model(self,model_path):
        self.model.save(path)
        print("Model saved to {path}")
        joblib.dump(self.scaler, "scaler.pkl")
    
    def load_model(self, model_path):
        self.model= load_model(path)
        print("Model loaded from {path}")
        self.scaler = joblib.load("scaler.pkl")
        
        
    def fit(self, train_data, y_train, weights_train=None):

        self.scaler.fit_transform(train_data)
        X_train = self.scaler.transform(train_data)
        self.model.fit(
            X_train, y_train, sample_weight=weights_train, epochs=5, verbose=2
        )
        self.model.save_model() 

    def predict(self, test_data):
        test_data = self.scaler.transform(test_data)
        return self.model.predict(test_data).flatten().ravel()
    
