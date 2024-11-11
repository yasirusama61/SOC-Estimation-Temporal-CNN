# SOC Estimation Using Temporal CNN

![SOC Estimation](https://img.shields.io/badge/SOC%20Estimation-Deep%20Learning-brightgreen)

This repository contains an implementation of a Temporal CNN model for State of Charge (SOC) estimation in lithium-ion batteries. The model demonstrates high accuracy with minimal prediction error, achieving a **Root Mean Squared Error (RMSE)** of **1.41%** in SOC estimation.

## 📂 Repository Structure

- **data/**: Data storage folder (not included in the repo).
- **src/**: Python scripts for data processing and model training.
- **notebooks/**: Jupyter notebooks for exploratory data analysis and testing.
- **results/**: Generated plots and model performance metrics.
- **README.md**: Project documentation.
- **requirements.txt**: List of required dependencies.

## 🚀 Data Processing

For SOC estimation, we used sequences of length 100, capturing the temporal dependencies in the SOC data over a fixed time window.

## 🔧 Model Architecture

Our final model is a **Temporal CNN** that includes:
- **Conv1D Layers**: For capturing temporal dependencies in the SOC sequences.
- **Dense Layers**: For final SOC prediction, with dropout and L2 regularization to minimize overfitting.

**Hyperparameters:**
- **Sequence Length**: 100
- **Embedding Dimension**: 64
- **Dropout Rate**: 0.4
- **Regularization**: L2 penalty

## ⚙️ Training and Evaluation

**Training Setup**:
- Optimizer: `Adam` with learning rate reduction on plateau.
- Loss Function: Mean Squared Error (MSE).
- Batch Size: 72.

**Performance Metrics**:
- **Mean Absolute Error (MAE)**: 0.0074
- **Mean Squared Error (MSE)**: 0.0002
- **Root Mean Squared Error (RMSE)**: 0.0141 (1.41% of SOC range)
- **R-squared (R²)**: 0.9978

## 📈 Results

![SOC Prediction Plot](results/soc_prediction_plot.png)

Our CNN-based model demonstrates a strong fit to the actual SOC values, with minimal deviation. The results show promise for accurate SOC estimation under various ambient conditions.

## 📜 Requirements

```plaintext
tensorflow==2.x
numpy
pandas
scipy
scikit-learn
matplotlib
plotly
```