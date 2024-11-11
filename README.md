# SOC Estimation Using Temporal CNN

![SOC Estimation](https://img.shields.io/badge/SOC%20Estimation-Deep%20Learning-brightgreen)

This repository contains an implementation of a Temporal CNN model for State of Charge (SOC) estimation in lithium-ion batteries. Previously, we explored LSTM (Long Short-Term Memory) models for SOC prediction tasks, given their effectiveness in handling sequential data. However, while LSTM performed well in many cases, we observed fluctuations in SOC predictions, particularly during rapid changes in SOC. These fluctuations motivated us to explore alternative architectures better suited for capturing temporal dependencies without introducing the level of complexity associated with LSTM models.

### Why Temporal CNN?
Temporal CNNs (Convolutional Neural Networks for time-series data) offer an efficient alternative to recurrent models like LSTM. Temporal CNNs can capture local temporal patterns in the data, and they are often faster and less prone to overfitting when dealing with large sequences. Unlike ConvLSTM, which combines convolution and recurrence, Temporal CNN focuses purely on convolutional layers to extract temporal features, making it both simpler and potentially more robust for our task. Through this approach, the Temporal CNN achieved precise SOC predictions with minimal fluctuations and a **Root Mean Squared Error (RMSE)** of **1.41%**, marking an improvement over prior methods.

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

![SOC Prediction Plot](results/soc_prediction_plot_0deg.png)

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

## 📈 Results

The Temporal CNN model for SOC estimation achieved high accuracy across various ambient temperatures, demonstrating its robustness in different operational conditions. Below are the detailed results:

- **0°C**: 
  - **Mean Absolute Error (MAE)**: 0.0074
  - **Mean Squared Error (MSE)**: 0.0002
  - **R-squared (R²)**: 0.9978

- **-10°C**:
  - **Mean Absolute Error (MAE)**: 0.0133
  - **Mean Squared Error (MSE)**: 0.0003
  - **R-squared (R²)**: 0.9946

- **10°C**:
  - **Mean Absolute Error (MAE)**: 0.0115
  - **Mean Squared Error (MSE)**: 0.00026
  - **R-squared (R²)**: 0.9966

- **25°C**:
  - **Mean Absolute Error (MAE)**: 0.0153
  - **Mean Squared Error (MSE)**: 0.0004
  - **R-squared (R²)**: 0.9953

### 🔍 Insights

1. **Consistent Performance Across Temperatures**:
   - The model demonstrates stable and reliable predictions across a range of temperatures, from -10°C to 25°C. This is a critical requirement in battery management systems, as SOC estimation must remain accurate across various environmental conditions.

2. **Accuracy at Low Temperatures**:
   - Even at lower temperatures (e.g., -10°C), where battery behavior can become more complex due to changes in electrochemical processes, the Temporal CNN maintained an **R-squared** value above 0.99. This shows the model's capability to adapt to the non-linearities in battery performance that emerge at low temperatures.

3. **Highest Precision at 0°C and 10°C**:
   - The model achieved its highest precision at 0°C and 10°C, with **MAE** values as low as 0.0074 and 0.0115, respectively, and **R-squared** values above 0.996. This suggests that the Temporal CNN is particularly effective in moderately low-temperature conditions, where SOC estimation tends to be challenging.

4. **Slight Decrease in Performance at Higher Temperatures**:
   - At 25°C, the model shows a slight increase in error metrics (**MAE** of 0.0153 and **R-squared** of 0.9953). While the performance remains high, this suggests that further fine-tuning or additional feature engineering (e.g., incorporating thermal dynamics) could improve model precision at higher temperatures.

5. **Overall Model Robustness**:
   - Achieving **RMSE percentages** below 1.5% for all temperatures demonstrates the model's robustness and suitability for SOC estimation across various environmental conditions, making it a reliable candidate for practical BMS applications.

These results highlight the efficacy of the Temporal CNN model in handling temperature variations, a significant factor in real-world battery applications.
