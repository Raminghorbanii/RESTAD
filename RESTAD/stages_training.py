#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from Utils import compute_and_update_centers, set_seed, get_encoder_output, hash_model_parameters
from LSTM_Model import LSTM
from Transformer_Model import Transformer, Transformer_RBF
from Training import Trainer
import torch


def pretrain_without_rbf(cfg, train_dataloader, test_dataloader, seed, save_mode = True):
    
    """
    Pre-trains the model without the Radial Basis Function (RBF) layer.
    
    This function handles the pre-training process of models without the RBF layer. 
    The pre-training is done using the model type specified in the configuration. 
    After training, the encoder's output is extracted, which can be used for 
    subsequent training phases. The function also provides an option to save 
    the encoder output and the trained model's state dictionary for future use.
    
    Args:
        cfg (object): Configuration object containing model and training settings.
        train_dataloader (DataLoader): DataLoader for the training dataset.
        test_dataloader (DataLoader): DataLoader for the test dataset.
        seed (int): Seed value for reproducibility.
        save_mode (bool, optional): If True, saves the encoder output and the trained model's 
                                    state dictionary. Defaults to True.
        
    Returns:
        trained_model (object): Model object after pre-training without the RBF layer.
    """

    print("####################")
    print(f"Starting pre-training without RBF using model type: {cfg.model.type}")
    
    base_model_name = cfg.model.base_model_type  # Name of the model used in the first phase

    # Set the seed for all relevant libraries
    set_seed(seed)
    print(f"Running experiment with seed {seed}")

    # Determine the model type based on the config
    if cfg.model.type == "LSTM":
        model = LSTM(cfg).to(cfg.device)
    if cfg.model.type == "Transformer":
        model = Transformer(cfg).to(cfg.device)

    print(model)

    trainer = Trainer(model, train_dataloader, test_dataloader, cfg, use_rbf=False)
    trained_model = trainer.train()
    print("Training completed.")
    print("####################")
    
    encoder_output = get_encoder_output(trained_model, train_dataloader, cfg.device)
    print("Encoder output extracted.")
    
    if save_mode:
        # Save the encoder output and the model's state dictionary for the next phase
        torch.save(encoder_output, f'{cfg.dataset.save_prefix}/encoder_output_{cfg.dataset.name}_{base_model_name}_{seed}_{cfg.initialization}_LocEnd.pth')
        torch.save(trained_model.state_dict(), f'{cfg.dataset.save_prefix}/trained_model_{cfg.dataset.name}_{base_model_name}_{seed}_{cfg.initialization}_LocEnd.pth')
            
    return trained_model





def train_with_rbf(cfg, train_dataloader, test_dataloader, seed, encoder_output=None, save_mode=True, load_model=False):
    """
    Trains or loads a model with the Radial Basis Function (RBF) layer.

    Depending on the load_model flag, this function either loads a pre-trained model
    or handles the training process of models that incorporate the RBF layer. 
    The function also provides an option to save the trained model's state dictionary 
    for future use.

    Args:
        cfg (object): Configuration object containing model and training settings.
        train_dataloader (DataLoader): DataLoader for the training dataset.
        test_dataloader (DataLoader): DataLoader for the test dataset.
        seed (int): Seed value for reproducibility.
        encoder_output (torch.Tensor, optional): Required for two-step initialization. Defaults to None.
        save_mode (bool, optional): If True, saves the trained model's state dictionary. Defaults to True.
        load_model (bool, optional): If True, loads the pre-trained model instead of training a new one. Defaults to False.

    Returns:
        rbf_model (object): The trained or loaded model object with the RBF layer.
    """

    print("####################")
    print(f"Starting training with RBF using model type: {cfg.model.type}")

    base_model_name = cfg.model.base_model_type
    set_seed(seed)
    print(f"Running experiment with seed {seed}")

    # Determine the model type and initialize or load model
    if cfg.model.type == "Transformer_RBF":
        rbf_model = Transformer_RBF(cfg).to(cfg.device)
        if load_model:
            model_path = f'{cfg.dataset.model_prefix}/trained_rbf_model_{cfg.dataset.name}_{base_model_name}_{seed}_{cfg.initialization}.pth'
            model_state_dict = torch.load(model_path, map_location=cfg.device)
            rbf_model.load_state_dict(model_state_dict)
            rbf_model.eval()
            print(f"Loaded model from {model_path}")
        else:
            if cfg.initialization == "2steps":
                print("Initialization method Activated: 2steps")
                compute_and_update_centers(encoder_output, rbf_model, cfg)
            elif cfg.initialization == "random":
                print("Initialization method Activated: random")

            model_hash = hash_model_parameters(rbf_model)
            print(f"Model hash: {model_hash}")

            trainer_rbf = Trainer(rbf_model, train_dataloader, test_dataloader, cfg, use_rbf=True)
            rbf_model = trainer_rbf.train()  # Ensure rbf_model is updated with the trained model
            print("Training completed.")

            if save_mode:
                torch.save(rbf_model.state_dict(), f'{cfg.dataset.save_prefix}/trained_rbf_model_{cfg.dataset.name}_{base_model_name}_{seed}_{cfg.initialization}.pth')
                print(f"Model saved at {cfg.dataset.save_prefix}")

    print("####################")

    return rbf_model




            
            