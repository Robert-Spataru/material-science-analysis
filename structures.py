from helpers import *
from model import *
import torch
import numpy as np

from helper import *
from dataclass import *

def get_data():
    hydrogen_array, nitrogen_array, energy_array, hydrogen_max_points, nitrogen_max_points = startup()
    lutetium_array = np.array(np.tile(get_lutetium_array(), (len(energy_array), 1)))
    return hydrogen_array, nitrogen_array, lutetium_array, energy_array

def get_nodes():
    hydrogen_array, nitrogen_array, lutetium_array, energy_array = get_data()
    all_stacked_nodes = []
    all_node_features = []
    for i in range(len(hydrogen_array)):
        hydrogen_indexed = hydrogen_array[i]
        hydrogen_length = len(hydrogen_indexed)
        nitrogen_indexed = nitrogen_array[i]
        nitrogen_length = len(nitrogen_indexed)
        lutetium_indexed = lutetium_array[i]
        lutetium_length = len(lutetium_indexed)
        stacked_nodes = np.vstack((hydrogen_indexed, nitrogen_indexed, lutetium_indexed))
        node_features = np.zeros((hydrogen_length + nitrogen_length + lutetium_length, 3))
        node_features[:hydrogen_length, 0] = 1
        node_features[hydrogen_length:hydrogen_length + nitrogen_length, 1] = 1
        node_features[hydrogen_length + nitrogen_length:, 2] = 1
        all_stacked_nodes.append(stacked_nodes)
        all_node_features.append(node_features)
    all_stacked_nodes = np.array(all_stacked_nodes)
    all_node_features = np.array(all_node_features)
    
    return all_stacked_nodes, all_node_features

def find_k_nearest_neighbors(all_stacked_nodes, all_node_features, k_neighbors):
    k_nearest_neighbors = []
    for i in range(len(all_stacked_nodes)):
        k_nearest_neighbors_of_each_element = []
        for j in range(len(all_stacked_nodes[i])):
            node = all_stacked_nodes[i][j]
            nodes = all_stacked_nodes[i]
            node_feature = all_node_features[i][j]
            node_features = all_node_features[i]
            distances = np.linalg.norm(nodes - node, axis=1)
            node_features_distances = {node_features[k]: distances[k] for k in range(len(node_features))}
            sorted_node_features_distances = sorted(node_features_distances.items(), key=lambda x: x[1])
            hydrogen_nearest = []
            nitrogen_nearest = []
            lutetium_nearest = []
            for k in range(len(sorted_node_features_distances)):
                if sorted_node_features_distances[k][0] == (1, 0, 0) and len(hydrogen_nearest) < k_neighbors:
                    
                    hydrogen_nearest.append(sorted_node_features_distances[k])
                elif sorted_node_features_distances[k][0] == (0, 1, 0) and len(nitrogen_nearest) < k_neighbors:
                    nitrogen_nearest.append(sorted_node_features_distances[k])
                elif sorted_node_features_distances[k][0] == (0, 0, 1) and len(lutetium_nearest) < k_neighbors:
                    lutetium_nearest.append(sorted_node_features_distances[k])
            k_nearest_neighbors_of_each_element.append(hydrogen_nearest + nitrogen_nearest + lutetium_nearest)

                
            
            
    

def get_edges(k_neighbors):
    pass


def main():
    pass