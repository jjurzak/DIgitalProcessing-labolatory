import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Generate white Gaussian noise
np.random.seed(123)
N = 1000
noise = np.random.normal(0, 1, N)

# Fit AR(1) model (order = (1,0,0))
model_ar1 = ARIMA(noise, order=(1, 0, 0))
result_ar1 = model_ar1.fit()
aic_ar1 = result_ar1.aic
resid_var_ar1 = np.var(result_ar1.resid)

# Fit AR(2) model (order = (2,0,0))
model_ar2 = ARIMA(noise, order=(2, 0, 0))
result_ar2 = model_ar2.fit()
aic_ar2 = result_ar2.aic
resid_var_ar2 = np.var(result_ar2.resid)

# Fit MA(1) model (order = (0,0,1))
model_ma1 = ARIMA(noise, order=(0, 0, 1))
result_ma1 = model_ma1.fit()
aic_ma1 = result_ma1.aic
resid_var_ma1 = np.var(result_ma1.resid)

# Compare AIC and Residual Variance
print("AR(1): AIC =", aic_ar1, "Residual Variance =", resid_var_ar1)
print("AR(2): AIC =", aic_ar2, "Residual Variance =", resid_var_ar2)
print("MA(1): AIC =", aic_ma1, "Residual Variance =", resid_var_ma1)

# Plot the original noise signal and AR(1) residuals as an example
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(noise, label="White Noise")
plt.title("White Gaussian Noise")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(result_ar1.resid, label="Residuals AR(1)", color='orange')
plt.title("Residuals of AR(1) Model")
plt.legend()

plt.tight_layout()
plt.show()

