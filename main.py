from helpers import *
from model import *
import torch
import pickle

    
def run_normal_neural_network(model_number):
    training_loss, validation_loss, test_metrics, best_val_loss, best_model_state = train_test_normal_neural_network(model_number)
    
    # Print model parameters
    for name, param in best_model_state.items():
        if param.requires_grad:
            print(name, param.data)
    
    print("Training Loss: ", training_loss)
    print("Validation Loss: ", validation_loss)
    print(f"Best Validation Loss: {best_val_loss}")
    print(f"Testing MSE: {test_metrics['mse']}")
    print(f"Testing MAE: {test_metrics['mae']}")
    print(f"Testing RMSE: {test_metrics['rmse']}")
    print(f"Correlation: {test_metrics['correlation']}")


def main():
    model_number_list = [0, 1]
    for model_number in model_number_list:
        run_normal_neural_network(model_number)
    #model = NeuralNetwork(input_size=216, hidden_size=256)
    #model.load_state_dict(torch.load(f'saved_models/neural_network_model.pth'{model_number}, weights_only=True))
    
main()