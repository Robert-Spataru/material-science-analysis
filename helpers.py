import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader
from dataclass import MoleculeDataset
import math
from model import *

def startup():
    hydrogen_pkl_path = 'data_used_pkl/hydrogen_matrices_numpy.pkl'
    nitrogen_pkl_path = 'data_used_pkl/nitrogen_matrices_numpy.pkl'
    energy_pkl_path = 'data_used_pkl/energy_column.pkl'
    hydrogen_array, nitrogen_array, energy_array = process_inputs(hydrogen_pkl_path, nitrogen_pkl_path, energy_pkl_path)
    print("Hydrogen Array Shape:", hydrogen_array.shape)
    print("Nitrogen Array Shape:", nitrogen_array.shape)
    print("Energy Array Shape:", energy_array.shape)
    hydrogen_max_points = get_max_points(hydrogen_array)
    nitrogen_max_points = get_max_points(nitrogen_array)
    print("Hydrogen Max Points:", hydrogen_max_points)
    print("Nitrogen Max Points:", nitrogen_max_points)
    return hydrogen_array, nitrogen_array, energy_array, hydrogen_max_points, nitrogen_max_points

def process_inputs(hydrogen_pkl_path, nitrogen_pkl_path, energy_pkl_path):
    with open(hydrogen_pkl_path, 'rb') as f:
        hydrogen_loader = pickle.load(f)
    with open(nitrogen_pkl_path, 'rb') as f:
        nitrogen_loader = pickle.load(f)
    with open(energy_pkl_path, 'rb') as f:
        energy_loader = pickle.load(f)
    hydrogen_array = np.array(hydrogen_loader)
    nitrogen_array = np.array(nitrogen_loader)
    energy_array = np.array(energy_loader)
    return hydrogen_array, nitrogen_array, energy_array

def get_element_list_length(element_array):
    element_lengths = []
    for i in range(len(element_array)):
        element_matrix = element_array[i]
        element_lengths.append(len(element_matrix))
    return element_lengths

def get_max_points(element_array):
    max_points = max(get_element_list_length(element_array))
    return max_points

def get_lutetium_array():
    lutetium_array = np.array([
    [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
    [0.4999999999999998, 0.0000000000000000, 0.0000000000000000],
    [0.0000000000000000, 0.4999999999999999, -0.0000000000000000],
    [0.0000000000000001, 0.0000000000000001, 0.4999999999999998],
    [0.4999999999999998, 0.4999999999999999, -0.0000000000000001],
    [0.5000000000000000, 0.0000000000000000, 0.4999999999999998],
    [0.0000000000000000, 0.5000000000000003, 0.4999999999999998],
    [0.4999999999999998, 0.5000000000000003, 0.4999999999999998]
    ])

    return lutetium_array

def pad_matrix(matrix, max_points):
    # Handle case where matrix might be empty or not the expected shape
    if len(matrix) == 0:
        # If matrix is empty, create an all-zero matrix of the requested size
        return np.zeros((max_points, 3))
    
    # Convert to numpy array if it's not already
    matrix = np.asarray(matrix)
    
    current_points = len(matrix)
    if current_points < max_points:
        # Create padding with zeros
        padding = np.zeros((max_points - current_points, 3))
        # Concatenate along first dimension
        padded_matrix = np.vstack((matrix, padding))
    else:
        # Truncate if too large
        padded_matrix = matrix[:max_points]
    
    return padded_matrix

def get_padded_matrices():

    hydrogen_array, nitrogen_array, energy_array, hydrogen_max_points, nitrogen_max_points = startup()
    lutetium_array = get_lutetium_array()
    
    # Add debugging for first few entries
    print("Sample hydrogen matrix shape:", hydrogen_array[0].shape if hasattr(hydrogen_array[0], 'shape') else "Unknown")
    print("Sample nitrogen matrix shape:", nitrogen_array[0].shape if hasattr(nitrogen_array[0], 'shape') else "Unknown")
    
    # Pad each matrix safely
    hydrogen_array_padded = np.array([pad_matrix(h_matrix, hydrogen_max_points) for h_matrix in hydrogen_array])
    nitrogen_array_padded = np.array([pad_matrix(n_matrix, nitrogen_max_points) for n_matrix in nitrogen_array])
    lutetium_array_padded = pad_matrix(lutetium_array, hydrogen_max_points)
    print("Padded Hydrogen Array Shape:", hydrogen_array_padded.shape)
    print("Padded Nitrogen Array Shape:", nitrogen_array_padded.shape)
    print("Lutetium Array Shape:", lutetium_array_padded.shape)
    return hydrogen_array_padded, nitrogen_array_padded, lutetium_array_padded, energy_array

def get_flattened_tensors():
    hydrogen_array, nitrogen_array, energy_array, hydrogen_max_points, nitrogen_max_points = startup()
    lutetium_array = get_lutetium_array()
    
    # Add debugging for first few entries
    print("Sample hydrogen matrix shape:", hydrogen_array[0].shape if hasattr(hydrogen_array[0], 'shape') else "Unknown")
    print("Sample nitrogen matrix shape:", nitrogen_array[0].shape if hasattr(nitrogen_array[0], 'shape') else "Unknown")
    
    # Pad each matrix safely
    hydrogen_array_padded = np.array([pad_matrix(h_matrix, hydrogen_max_points) for h_matrix in hydrogen_array])
    nitrogen_array_padded = np.array([pad_matrix(n_matrix, nitrogen_max_points) for n_matrix in nitrogen_array])
    lutetium_array_padded = pad_matrix(lutetium_array, hydrogen_max_points)
    print("Padded Hydrogen Array Shape:", hydrogen_array_padded.shape)
    print("Padded Nitrogen Array Shape:", nitrogen_array_padded.shape)
    print("Lutetium Array Shape:", lutetium_array_padded.shape)
    
    
    flattened_hydrogen_array = np.array([matrix.flatten() for matrix in hydrogen_array_padded])
    flattened_nitrogen_array = np.array([matrix.flatten() for matrix in nitrogen_array_padded])
    flattened_lutetium_array = np.array(np.tile(lutetium_array_padded.flatten(), (len(hydrogen_array), 1)))
    print("Flattened Hydrogen Array Shape:", flattened_hydrogen_array.shape)
    print("Flattened Nitrogen Array Shape:", flattened_nitrogen_array.shape)
    print("Flattened Lutetium Array Shape (Repeated):", flattened_lutetium_array.shape)
    
    # Concatenate the flattened arrays
    combined_array = np.concatenate((flattened_hydrogen_array, flattened_nitrogen_array, flattened_lutetium_array), axis=1)
    print("Combined Array Shape:", combined_array.shape)
    # Convert to PyTorch tensor
    combined_tensor = torch.tensor(combined_array, dtype=torch.float32)
    print("Combined Tensor Shape:", combined_tensor.shape)
    # Convert energy array to PyTorch tensor
    energy_tensor = torch.tensor(energy_array, dtype=torch.float32)
    print("Energy Tensor Shape:", energy_tensor.shape)
    return combined_tensor, energy_tensor
    
def train_test_normal_neural_network(model_number):
    #combined_tensor, energy_tensor = get_flattened_tensors()
    from distance_array import create_distance_array
    combined_tensor, energy_tensor = create_distance_array()
    # Create a dataset
    dataset = MoleculeDataset(combined_tensor, energy_tensor)
    
    # Split the dataset into training, validation, testing sets
    training_size = int(0.8 * len(dataset))
    validation_size = int(0.1 * len(dataset))
    testing_size = len(dataset) - training_size - validation_size
    training_dataset, validation_dataset, testing_dataset = random_split(dataset, [training_size, validation_size, testing_size])
    print(f"Training dataset size: {len(training_dataset)}")
    print(f"Validation dataset size: {len(validation_dataset)}")
    print(f"Testing dataset size: {len(testing_dataset)}")  
    
    # Create DataLoaders
    training_loader = DataLoader(training_dataset, batch_size=32, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
    testing_loader = DataLoader(testing_dataset, batch_size=32, shuffle=False)
    print(f"Training DataLoader size: {len(training_loader)}")
    print(f"Validation DataLoader size: {len(validation_loader)}")
    print(f"Testing DataLoader size: {len(testing_loader)}")
    
    input_size = combined_tensor.shape[1]
    hidden_size = 256
    if model_number == 0:
        model_neural_network = NeuralNetwork(input_size, hidden_size, output_size=1)
        new_model_neural_network = NeuralNetwork(input_size, hidden_size, output_size=1)
    elif model_number == 1:
        model_neural_network = OptimizedNeuralNetwork(input_size, hidden_size, output_size=1)
        new_model_neural_network = OptimizedNeuralNetwork(input_size, hidden_size, output_size=1)
    else:
        raise ValueError("Invalid model number. Choose 0, 1, or 2.")
    print(f"Model Number: {model_number}")
    optimizer = torch.optim.Adam(model_neural_network.parameters(), lr=0.001)

    #criterion = torch.nn.MSELoss()
    criterion = nn.SmoothL1Loss()
    results = training_loop(model_neural_network, training_loader, validation_loader, optimizer, criterion, epochs=10)
    training_loss = results['training_loss']
    validation_loss = results['validation_loss']
    best_val_loss = results['best_val_loss']
    best_model_state = results['best_model_state']
    
    #save_model_state(best_model_state, f'saved_models/neural_network_model{model_number}.pth')
    

    new_model_neural_network.load_state_dict(best_model_state)
    test_metrics = testing(new_model_neural_network, testing_loader, criterion, model_number)
    
    return training_loss, validation_loss, test_metrics, best_val_loss, best_model_state
    

def plot_num_atoms_energy(hydrogen_array, nitrogen_array, energy_array):
    hydrogen_lengths = get_element_list_length(hydrogen_array)
    nitrogen_lengths = get_element_list_length(nitrogen_array)
    energy_values = energy_array.flatten()
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(projection='3d')
    ax.scatter3D(hydrogen_lengths, nitrogen_lengths, energy_values, c=energy_values, cmap='viridis', marker='o')
    ax.set_xlabel('# of Hydrogen Atoms')
    ax.set_ylabel('# of Nitrogen Atoms')
    ax.set_zlabel('Energy Values')
    plt.title('3D Scatter Plot of Hydrogen and Nitrogen Lengths vs Energy')
    plt.show()
    plt.savefig('saved_plots/3d_scatter_plot.png')

def training_loop(model, train_loader, val_loader, optimizer, criterion, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    running_training_loss = []
    running_validation_loss = []
    
    # Initialize variables to track best model
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        current_training_loss = 0.0
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1)
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            current_training_loss += loss.item()  # Fixed typo in variable name
        
        avg_train_loss = current_training_loss/len(train_loader)
        running_training_loss.append(avg_train_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_train_loss}")

        # Validation
        model.eval()
        current_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                current_val_loss += loss.item()
        scheduler.step(current_val_loss)
        avg_val_loss = current_val_loss/len(val_loader)
        running_validation_loss.append(avg_val_loss)
        print(f"Validation Loss: {avg_val_loss}")
        
        # Check if this is the best model so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            print(f"New best model saved with validation loss: {best_val_loss}")
    
    # Return the training history and best model state
    return {
        'training_loss': running_training_loss,
        'validation_loss': running_validation_loss,
        'best_val_loss': best_val_loss,
        'best_model_state': best_model_state
    }

def testing(model, testing_loader, criterion, model_number):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in testing_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # Calculate MSE loss
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)  # Multiply by batch size
            
            # Calculate MAE
            mae = torch.nn.functional.l1_loss(outputs, targets)
            total_mae += mae.item() * inputs.size(0)
            
            # Store predictions and targets for more detailed analysis
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            total_samples += inputs.size(0)
    
    # Calculate average losses
    avg_mse = total_loss / total_samples
    avg_mae = total_mae / total_samples
    
    print(f"Testing MSE Loss: {avg_mse:.6f}")
    print(f"Testing MAE Loss: {avg_mae:.6f}")
    print(f"Testing RMSE: {math.sqrt(avg_mse):.6f}")
    
    # You can also calculate correlation coefficient
    predictions_array = np.array(all_predictions).flatten()
    targets_array = np.array(all_targets).flatten()
    correlation = np.corrcoef(predictions_array, targets_array)[0, 1]
    print(f"Pearson Correlation: {correlation:.6f}")
    
    # Optional: Plot predicted vs actual values
    plt.figure(figsize=(8, 8))
    plt.scatter(targets_array, predictions_array, alpha=0.5)
    plt.plot([min(targets_array), max(targets_array)], [min(targets_array), max(targets_array)], 'r--')
    plt.xlabel('Actual Energy')
    plt.ylabel('Predicted Energy')
    plt.title('Predicted vs Actual Energy Values')
    plt.savefig(f'saved_plots/prediction_vs_actual_distances{model_number}.png')
    plt.close()
    
    plt.figure(figsize=(8, 8))
    indices = np.arange(len(targets_array))
    plt.scatter(targets_array, abs(targets_array - predictions_array), alpha=0.5)
    plt.xlabel('Actual Value')
    plt.ylabel('Absolute Error')
    plt.title('Absolute Error of Predictions')
    plt.savefig(f'saved_plots/absolute_error_actual_distances{model_number}.png')
    plt.close()
    
    return {
        'mse': avg_mse, 
        'mae': avg_mae, 
        'rmse': math.sqrt(avg_mse),
        'correlation': correlation,
        'predictions': predictions_array,
        'targets': targets_array
    }

def save_model_state(model_state, path):
    torch.save(model_state, path)

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))