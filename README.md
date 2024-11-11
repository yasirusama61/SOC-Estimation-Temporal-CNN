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

## 🏗️ Model Architecture

The model architecture for SOC estimation is designed as a Temporal CNN, leveraging a series of convolutional layers to capture temporal dependencies in the SOC data. This approach allows the model to learn from sequences without the need for recurrent layers, providing efficient training and accurate predictions.

### 🔹 Architecture Overview

1. **Input Layer**: 
   - Accepts input sequences with a shape of `(100, num_features)`, where `100` is the sequence length.

2. **First Convolutional Block**
   - **Conv1D Layer**: 64 filters, kernel size of 3, ReLU activation.
   - **Batch Normalization**: Helps stabilize the learning process.
   - **Dropout**: 30% to prevent overfitting.
   - **Purpose**: Captures the initial temporal dependencies in the SOC data.

3. **Second Convolutional Block**
   - **Conv1D Layer**: 128 filters, kernel size of 5, ReLU activation.
   - **Batch Normalization**: Ensures stable gradients and faster convergence.
   - **Dropout**: 30% for regularization.
   - **Purpose**: Expands on the temporal patterns identified in the first block, capturing more intricate relationships.

4. **Third Convolutional Block**
   - **Conv1D Layer**: 256 filters, kernel size of 5, ReLU activation.
   - **Batch Normalization** and **Dropout (40%)**: Helps maintain model stability and reduces overfitting.
   - **Purpose**: Allows the model to learn deeper temporal features, critical for accurately predicting SOC over longer sequences.

5. **Global Average Pooling Layer**
   - **GlobalAveragePooling1D**: Reduces data dimensionality while retaining essential information.
   - **Purpose**: Summarizes the most relevant temporal features from each filter for final dense layers.

6. **Fully Connected Dense Layers**
   - **Dense Layer**: 64 units with ReLU activation and L2 regularization.
   - **Dropout**: 40% for robust regularization.
   - **Purpose**: Learns high-level representations based on the pooled features, ensuring good generalization.

7. **Output Layer**
   - **Dense Layer**: 1 unit with linear activation for SOC regression.
   - **Purpose**: Provides the final SOC estimation, outputting a continuous value representing the predicted SOC.

---

### 🤖 Why Temporal CNN?

The Temporal CNN is chosen for its efficiency in handling sequential data without the complexity of recurrent connections. By stacking convolutional layers, the model captures temporal dependencies and local patterns, making it particularly effective for SOC data with fluctuating trends across different temperatures. Regularization techniques such as dropout and L2 regularization ensure robustness, reducing overfitting even with complex SOC patterns.

### 📊 Model Summary

Below is a summary of the model architecture, detailing the output shapes and parameters for each layer:

![Model Summary](results/model_summary.png)

---

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

The Temporal CNN model demonstrated exceptional performance across various temperature conditions, maintaining high accuracy and minimal error. Key results for each temperature setting are summarized below:

| Temperature | Mean Absolute Error (MAE %) | Root Mean Squared Error (RMSE %) | R-squared (R²) |
|-------------|-----------------------------|----------------------------------|-----------------|
| -10°C       | 1.33%                       | 1.82%                            | 0.9946         |
| 0°C         | 0.74%                       | 1.41%                            | 0.9978         |
| 10°C        | 1.15%                       | 1.60%                            | 0.9966         |
| 25°C        | 1.53%                       | 2.00%                            | 0.9953         |

### 🔍 Insights

- **Consistency Across Temperatures**: The model achieved high R² scores (above 0.99) across all tested temperatures, showcasing its robustness and suitability for diverse environmental conditions.
- **Reduced SOC Fluctuations**: Compared to previous LSTM-based models, the Temporal CNN significantly minimized prediction fluctuations, especially in challenging conditions like -10°C, providing stable and reliable SOC estimates.
- **Practical Implications**: With low error rates across varying temperatures, this model is highly applicable to real-world Battery Management Systems (BMS) in electric vehicles, energy storage, and other applications where precise SOC monitoring is essential.

The visualizations below illustrate the close alignment between actual and predicted SOC across all tested temperatures, emphasizing the model’s accuracy and stability.

![SOC Prediction at 0°C](results/soc_prediction_plot_0deg.png)
![SOC Prediction at -10°C](results/soc_prediction_plot_-10deg.png)
![SOC Prediction at 10°C](results/soc_prediction_plot_10deg.png)
![SOC Prediction at 25°C](results/soc_prediction_plot_25deg.png)
