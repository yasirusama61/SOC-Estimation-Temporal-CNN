#!/usr/bin/env python
# coding: utf-8

"""
SOC Estimation CNN Model
Author: Usama Yasir Khan
Description: This script loads SOC estimation data, processes sequences for ConvLSTM and CNN models, trains and evaluates models, and generates plots for SOC predictions.
"""

# Import necessary libraries
import scipy.io
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import ConvLSTM2D, Conv1D, Dense, Dropout, Flatten, BatchNormalization, GlobalAveragePooling1D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Data loading and processing functions
def load_mat_file(file_path, input_columns, target_column):
    """Load data from a .mat file and return input and target dataframes."""
    mat_file = scipy.io.loadmat(file_path)
    X = mat_file['X'].T
    Y = mat_file['Y'].T
    df_X = pd.DataFrame(X, columns=input_columns)
    df_Y = pd.DataFrame(Y, columns=[target_column])
    return pd.concat([df_X, df_Y], axis=1)

def create_sequences(X, y, timesteps):
    """Create sequences from data for training and validation."""
    X_seq, y_seq = [], []
    for i in range(len(X) - timesteps):
        X_seq.append(X[i:i + timesteps])
        y_seq.append(y[i + timesteps])
    return np.array(X_seq), np.array(y_seq)

# Load and process data
train_file = 'TRAIN_LGHG2@n10degC_to_25degC_Norm_5Inputs.mat'
input_columns = ['Voltage', 'Current', 'Temperature', 'Avg_voltage', 'Avg_current']
target_column = 'SOC'
df_train = load_mat_file(train_file, input_columns, target_column)
X_train, y_train = df_train[input_columns], df_train[target_column]

validation_file = '01_TEST_LGHG2@n10degC_Norm_(05_Inputs).mat'
df_val = load_mat_file(validation_file, input_columns, target_column)
X_val, y_val = df_val[input_columns], df_val[target_column]

# Sequence generation
timesteps = 30
X_train_seq, y_train_seq = create_sequences(X_train, y_train, timesteps)
X_val_seq, y_val_seq = create_sequences(X_val, y_val, timesteps)

# Reshape data for ConvLSTM2D model
X_train_conv = X_train_seq.reshape((X_train_seq.shape[0], X_train_seq.shape[1], 1, 1, X_train_seq.shape[2]))
X_val_conv = X_val_seq.reshape((X_val_seq.shape[0], X_val_seq.shape[1], 1, 1, X_val_seq.shape[2]))
y_train_conv, y_val_conv = y_train_seq, y_val_seq

# ConvLSTM Model Definition
def build_convlstm_model(input_shape):
    model = Sequential()
    model.add(ConvLSTM2D(32, (1, 2), activation='relu', input_shape=input_shape, return_sequences=True, padding='same'))
    model.add(Dropout(0.3))
    model.add(ConvLSTM2D(32, (1, 2), activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(20, activation='relu', kernel_regularizer='l2'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

convlstm_model = build_convlstm_model((timesteps, 1, 1, X_train_seq.shape[2]))
convlstm_model.summary()

# Callbacks for training
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
model_checkpoint = ModelCheckpoint('convolstm_model.keras', save_best_only=True, monitor='val_loss', mode='min')

# Train ConvLSTM model
history_convlstm = convlstm_model.fit(
    X_train_conv, y_train_conv,
    validation_data=(X_val_conv, y_val_conv),
    epochs=100, batch_size=72,
    callbacks=[early_stopping, reduce_lr, model_checkpoint]
)

# Temporal CNN Model Definition
def build_temporal_cnn_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        Dropout(0.3),
        Conv1D(128, kernel_size=5, activation='relu', padding='same'),
        BatchNormalization(),
        Dropout(0.3),
        Conv1D(256, kernel_size=5, activation='relu', padding='same'),
        BatchNormalization(),
        Dropout(0.4),
        GlobalAveragePooling1D(),
        Dense(64, activation='relu', kernel_regularizer='l2'),
        Dropout(0.4),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005), loss="mse", metrics=["mae"])
    return model

cnn_model = build_temporal_cnn_model((100, X_train_seq.shape[2]))
cnn_model.summary()

# Train Temporal CNN model
history_cnn = cnn_model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_val_seq, y_val_seq),
    epochs=50, batch_size=256,
    callbacks=[early_stopping, reduce_lr, model_checkpoint]
)

# Function to plot training history
def plot_training_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# Call the function to plot the training history
plot_training_history(history_cnn)

# Load and process test data
def load_test_data(test_file_path, input_columns, target_column):
    df_test = load_mat_file(test_file_path, input_columns, target_column)
    X_test = df_test[input_columns]
    y_test = df_test[target_column]
    return create_sequences(X_test.values, y_test.values, timesteps)

test_file = 'Test/02_TEST_LGHG2@0degC_Norm_(05_Inputs).mat'
X_test_seq, y_test_seq = load_test_data(test_file, input_columns, target_column)

# Evaluate the model and plot predictions
y_pred = cnn_model.predict(X_test_seq)
mae = mean_absolute_error(y_test_seq, y_pred)
rmse = mean_squared_error(y_test_seq, y_pred, squared=False)
r2 = r2_score(y_test_seq, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")

# Plot Actual vs Predicted SOC
plt.figure(figsize=(12, 6))
plt.plot(y_test_seq, label='Actual SOC', color='blue')
plt.plot(y_pred, label='Predicted SOC', color='red', linestyle='--')
plt.title('Actual vs Predicted SOC')
plt.xlabel('Samples')
plt.ylabel('SOC')
plt.legend()
plt.grid(True)
plt.show()

# Residual Plot
def plot_residuals(y_test, y_pred, temp_label):
    residuals = y_test - y_pred.flatten()
    plt.figure(figsize=(12, 6))
    plt.scatter(y_pred, residuals, alpha=0.5, color='purple')
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel('Predicted SOC')
    plt.ylabel('Residuals')
    plt.title(f'Residuals vs Predicted SOC for {temp_label}')
    plt.grid(True)
    plt.show()

# Plot residuals for the test set
plot_residuals(y_test_seq, y_pred, '0°C')
