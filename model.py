# Gabungan kode program prediksi curah hujan dengan Streamlit
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import streamlit as st
import matplotlib.pyplot as plt

# Langkah 1: Memuat dan Memeriksa Data
file_path = 'data.csv'  # Sesuaikan dengan path file dataset Anda
data = pd.read_csv(file_path, delimiter=';')

print("Pratinjau Data:")
print(data.head())

print("\nInformasi Data:")
print(data.info())

print("\nStatistik Deskriptif:")
print(data.describe())

# Langkah 2: Pemilihan Fitur dan Target
features = ['Tn', 'Tx', 'Tavg', 'RH_avg', 'ss', 'ff_x', 'ddd_x', 'ff_avg']
target = 'RR'

# Langkah 3: Standarisasi dan Transformasi Data
X = data[features]
y = data[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_scaled)

# Langkah 4: Pemisahan Data dan Pembuatan Model dengan GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

gbr = GradientBoostingRegressor(random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Prediksi dan Evaluasi
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"MAE pada data training: {mae_train:.2f}")
print(f"MAE pada data testing: {mae_test:.2f}")
print(f"R² pada data training: {r2_train:.2f}")
print(f"R² pada data testing: {r2_test:.2f}")

# Menyimpan model dan scaler
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(poly, 'poly.pkl')
joblib.dump(features, 'features.pkl')

# Langkah 5: Fungsi Evaluasi dan Plot Hasil Prediksi
def plot_results(y_true, y_pred, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', lw=2, linestyle='--')
    plt.xlabel('Actual Rainfall')
    plt.ylabel('Predicted Rainfall')
    plt.title(title)
    plt.show()

plot_results(y_train, y_train_pred, "Actual vs Predicted Rainfall (Training Data)")
plot_results(y_test, y_test_pred, "Actual vs Predicted Rainfall (Testing Data)")

