import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from sklearn import metrics

sheffield_dataframe = pd.read_csv('Sheffield Collision Data Cleaned.csv')

# SANITY CHECKING
print("")
print(sheffield_dataframe.shape)    #   Displaying the number of columns and rows within the dataset.
print(sheffield_dataframe.isnull().sum())   #   How many items within each column are null? - result shows none.
print("")

#       PAIR PLOT

#sbn.pairplot(sheffield_dataframe, plot_kws={'color': 'orange'})

#       CORRELATION HEATMAP

x = sheffield_dataframe.select_dtypes(include=["number"])

x = x.drop(columns=["collision_adjusted_severity_serious","collision_adjusted_severity_slight"],
    errors="ignore"
)   #Dropping the items present within the column that I will use as the label.

y =sheffield_dataframe['collision_adjusted_severity_serious'].map({"Not serious": 0, "Serious": 1}) #As an integer so it gives a numeric value
print(y.isna().sum())   #Ensuring none of the y values are null.

#Displaying a Correlation Heatmap (based on the x values entered above)
corr_matrix = x.corr()

plt.figure(figsize=(14, 14))
sbn.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.show()

#       Strongly Correlated

#   collision_year & collision_injury_based
#   enhanced_severity_collision & collision_injury_based
#   attend_scene_of_accident & enhanced_severity_collision
#   collision_year & collision_injury_based

x = sheffield_dataframe[['collision_year']]
y = sheffield_dataframe[['collision_injury_based']]

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=4)

linear_regression_model = LinearRegression()

linear_regression_model.fit(x_train, y_train)

predictions = x_test

plt.figure(figsize=(8, 6))

# Scatter plot: Actual vs Predicted values
plt.scatter(y_test, predictions, edgecolor='black', alpha=0.7, color='plum', label='Predicted Points')

# Regression line (best fit) through predicted vs actual values
z = np.polyfit(y_test, predictions, 1)  # Linear fit (degree=1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), color='red', linewidth=2, label='Regression Line')

# Perfect prediction line (y=x)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--', color='green', linewidth=2, label='Perfect Prediction')

# Labels and Title
plt.xlabel('Actual collision year (Y Test)', fontsize=12, weight='bold')
plt.ylabel('Predicted Collision Year (Y Pred)', fontsize=12, weight='bold')
plt.title('Actual vs Predicted House Prices with Regression Line', fontsize=14, weight='bold')

plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.show()

#   PLOT COST FUNCTION

# --- Assume X_test, y_test, predictions exist ---
X = x_test.to_numpy().reshape(-1)
y = y_test.to_numpy().reshape(-1)

# --- Normalize X and y for stable plotting ---
X_mean, X_std = X.mean(), X.std()
y_mean, y_std = y.mean(), y.std()
X_norm = (X - X_mean) / X_std
y_norm = (y - y_mean) / y_std

# --- Assume predictions exist ---
pred = predictions.reshape(-1)
pred_norm = (pred - y_mean) / y_std

# --- Create small grid around predictions for visualization ---
# We'll simulate perturbing predictions slightly to see effect on MAE/MSE/RMSE
delta = 0.1  # small variation
theta0_vals = np.linspace(-delta, delta, 50)  # small intercept shift
theta1_vals = np.linspace(0.9, 1.1, 50)      # small slope multiplier
T0, T1 = np.meshgrid(theta0_vals, theta1_vals)

# --- Initialize metric surfaces ---
MAE_surface = np.zeros(T0.shape)
MSE_surface = np.zeros(T0.shape)
RMSE_surface = np.zeros(T0.shape)

# --- Compute metrics for perturbed predictions ---
for i in range(T0.shape[0]):
    for j in range(T0.shape[1]):
        y_pred = T0[i, j] + T1[i, j] * pred_norm  # perturb predictions
        MSE_surface[i, j] = metrics.mean_squared_error(y_norm, y_pred)
        MAE_surface[i, j] = metrics.mean_absolute_error(y_norm, y_pred)
        RMSE_surface[i, j] = np.sqrt(metrics.mean_squared_error(y_norm, y_pred))

# --- Plot 3D surfaces ---
fig = plt.figure(figsize=(22, 6))

# ---- MAE subplot ----
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
surf1 = ax1.plot_surface(T0, T1, MAE_surface, cmap='plasma', alpha=0.9, edgecolor='k', linewidth=0.2)
ax1.set_xlabel("Theta0 Shift", labelpad=12)
ax1.set_ylabel("Theta1 Multiplier", labelpad=12)
ax1.set_zlabel("MAE", labelpad=15)
ax1.set_title("MAE Surface", pad=20)
ax1.view_init(elev=35, azim=135)
ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax1.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, pad=0.1).set_label('MAE')

# ---- MSE subplot ----
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
surf2 = ax2.plot_surface(T0, T1, MSE_surface, cmap='viridis', alpha=0.9, edgecolor='k', linewidth=0.2)
ax2.set_xlabel("Theta0 Shift", labelpad=12)
ax2.set_ylabel("Theta1 Multiplier", labelpad=12)
ax2.set_zlabel("MSE", labelpad=15)
ax2.set_title("MSE Surface", pad=20)
ax2.view_init(elev=35, azim=135)
ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax2.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, pad=0.1).set_label('MSE')

# ---- RMSE subplot ----
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
surf3 = ax3.plot_surface(T0, T1, RMSE_surface, cmap='cividis', alpha=0.9, edgecolor='k', linewidth=0.2)
ax3.set_xlabel("Theta0 Shift", labelpad=12)
ax3.set_ylabel("Theta1 Multiplier", labelpad=12)
ax3.set_zlabel("RMSE", labelpad=15)
ax3.set_title("RMSE Surface", pad=20)
ax3.view_init(elev=35, azim=135)
ax3.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax3.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10, pad=0.1).set_label('RMSE')

plt.tight_layout(w_pad=3)
plt.show()

#   Predicting based on user input

avg_income = float(input("Enter Avg. Area Income(in $): "))

# Create a DataFrame with the correct column name for prediction
input_features_df = pd.DataFrame({'Avg. Area Income': [avg_income]})

# Predict the house price
predicted_price = predict(input_features_df)

print(f"\n🔎 Predicted House Price: ${predicted_price[0]:,.2f}")

