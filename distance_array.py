from model import *
import torch
import pickle
import numpy as np

#each element should be a 1D vector of size 1 x 24
def get_element_lutetium_distances_array(element_array_padded, lutetium_array_padded):
    element_lutetium_distances_array = []
    for i in range(len(element_array_padded)):
        temp_distances = []
        for j in range(len(lutetium_array_padded)):
            if not np.array_equal(element_array_padded[i], np.zeros(3)) and not np.array_equal(lutetium_array_padded[j], np.zeros(3)):
                element_lutetium_distance = np.linalg.norm(element_array_padded[i] - lutetium_array_padded[j])
                temp_distances.append(element_lutetium_distance)
            elif np.array_equal(element_array_padded[i], np.zeros(3)) or np.array_equal(lutetium_array_padded[j], np.zeros(3)):
                element_lutetium_distance = np.inf
                temp_distances.append(element_lutetium_distance)
        element_lutetium_distances = np.array(temp_distances)
        minimum_distance = np.min(element_lutetium_distances)
        element_lutetium_distances_array.append(minimum_distance)
    return np.array(element_lutetium_distances_array)

#each element should be a 2D array of size 24 x 24
def get_hydrogen_nitrogen_distances_array(hydrogen_array_padded, nitrogen_array_padded):
    hydrogen_nitrogen_distances_array = []
    for i in range(len(hydrogen_array_padded)):
        temp_distances = []
        for j in range(len(nitrogen_array_padded)):
            if not np.array_equal(hydrogen_array_padded[i], np.zeros(3)) and not np.array_equal(nitrogen_array_padded[j], np.zeros(3)):
                hydrogen_nitrogen_distance = np.linalg.norm(hydrogen_array_padded[i] - nitrogen_array_padded[j])
                temp_distances.append(hydrogen_nitrogen_distance)
            elif np.array_equal(hydrogen_array_padded[i], np.zeros(3)) or np.array_equal(nitrogen_array_padded[j], np.zeros(3)):
                hydrogen_nitrogen_distance = np.inf
                temp_distances.append(hydrogen_nitrogen_distance)
        hydrogen_nitrogen_distances = np.array(temp_distances)
        hydrogen_nitrogen_distances_array.append(hydrogen_nitrogen_distances)
    return np.array(hydrogen_nitrogen_distances_array)

def create_distance_array():
    from helpers import get_padded_matrices
    hydrogen_array_padded, nitrogen_array_padded, lutetium_array_padded, energy_array = get_padded_matrices()
    hydrogen_lutetium_distances_array = get_element_lutetium_distances_array(hydrogen_array_padded, lutetium_array_padded)
    nitrogen_lutium_distances_array = get_element_lutetium_distances_array(nitrogen_array_padded, lutetium_array_padded)
    hydrogen_nitrogen_distances_array = get_hydrogen_nitrogen_distances_array(hydrogen_array_padded, nitrogen_array_padded)
    flattened_array = []
    for i in range(len(hydrogen_array_padded)):
        h_lu_distance = np.array([hydrogen_lutetium_distances_array[i]])
        n_lu_distance = np.array([nitrogen_lutium_distances_array[i]])
        h_n_distance = np.array([hydrogen_nitrogen_distances_array[i]])
        flattened_array.append(np.concatenate((h_lu_distance.flatten(), n_lu_distance.flatten(), h_n_distance.flatten()), axis=0))
    flattened_array = np.array(flattened_array)
    return torch.tensor(flattened_array, dtype=torch.float32), torch.tensor(energy_array, dtype=torch.float32)

    
    