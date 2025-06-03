import pickle
import matplotlib.pyplot as plt
import pandas as pd

file_path = 'blackSwan_data/blackSwan_data.parquet'
data = pd.read_parquet(file_path)


# Charger le modèle
import joblib

model = joblib.load("scalers/scaler.pkl")

import xgboost as xgb

# Si model est un Booster XGBoost ou un wrapper sklearn
model.get_booster().get_score(importance_type='weight')


plt.bar(data.columns.values, xgb.feature_importances_)
plt.xticks(rotation=90)
plt.title("Feature importances XGBoost Hist")
#plt.savefig(new_dir + "/VarImp_BDT_XGBoost_Hist.pdf",bbox_inches='tight')
plt.show()
plt.bar(data.columns.values, gbm.feature_importances_)
plt.xticks(rotation=90)
plt.title("Feature importances LightGBM")
#plt.savefig(new_dir + "/VarImp_BDT_LightGBM.pdf",bbox_inches='tight')
plt.show()