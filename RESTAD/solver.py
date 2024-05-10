#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from stages_training import pretrain_without_rbf, train_with_rbf
from evaluation import Reconstruction_Errors_Clculation, evaluate_based_on_reconstruction, evaluate_based_on_SimRec
import torch



def run_experiment(cfg, train_dataloader, test_dataloader, seed, save_mode, load_model):

    """
    Run the experiment based on the provided configuration, data loaders, and seed.
    
    Args:
        cfg: Configuration object containing necessary parameters.
        train_dataloader: DataLoader object containing the training dataset.
        test_dataloader: DataLoader object containing the test dataset.
        seed: Seed value for reproducibility.
        
    Returns:
        dict: Results of the experiment for the given seed.
    """
    
    results = {}

    print(f"Selected experiment with seed: {seed}")
    print(f"Initialization method: {cfg.initialization}")
    
    # Handle the "2steps" initialization method
    if cfg.initialization == "2steps":
        
        print(f"Mode: {cfg.mode}")
        
        # Pre-training phase
        if cfg.mode == "pretrain":
            
            print("Starting pre-training phase...")
            
            trained_model = pretrain_without_rbf(cfg, train_dataloader, test_dataloader, seed, save_mode = save_mode)
            reconstruction_errors_point_tr, reconstruction_errors_point_te, true_labels_point = Reconstruction_Errors_Clculation(trained_model, train_dataloader, test_dataloader, cfg)
            Rec_point_wise_results = evaluate_based_on_reconstruction(reconstruction_errors_point_tr, reconstruction_errors_point_te, true_labels_point, cfg)
            results[seed] = {'Rec (preTrain) Results': Rec_point_wise_results} # Save the results for this seed
            
            print("Pre-training phase completed.")
            
        # Main training phase
        elif cfg.mode == "main_train":
    
            print("Starting main training phase...")
            
            base_model_name = cfg.model.base_model_type  # Name of the model used in the first phase
            
            # Load the encoder output and the model's state dictionary from the previous phase
            pretrain_encoder_output = torch.load(f'{cfg.dataset.model_prefix}/encoder_output_{cfg.dataset.name}_{base_model_name}_{seed}_{cfg.initialization}.pth')
            
            trained_rbf_model = train_with_rbf(cfg, train_dataloader, test_dataloader, seed, encoder_output=pretrain_encoder_output, save_mode = save_mode, load_model=load_model)
            reconstruction_errors_point_tr, reconstruction_errors_point_te, true_labels_point = Reconstruction_Errors_Clculation(trained_rbf_model, train_dataloader, test_dataloader, cfg)
            Rec_point_wise_results = evaluate_based_on_reconstruction(reconstruction_errors_point_tr, reconstruction_errors_point_te, true_labels_point, cfg)
            SimRec_point_wise_results, SimRec_Addition_point_wise_results = evaluate_based_on_SimRec(trained_rbf_model, train_dataloader, test_dataloader, reconstruction_errors_point_tr, reconstruction_errors_point_te, true_labels_point, cfg)
            
            # Save the results for this seed
            results[seed] = {'SimRec Results': SimRec_point_wise_results,
                             'Rec Results Results': Rec_point_wise_results,
                             'SimRec Addtion Results': SimRec_Addition_point_wise_results}
            
            print("Main training phase completed.")
            
        else:
            raise ValueError("Invalid mode. Phase use the correct mode for training")
    
    
    # Handle the "random" initialization method
    elif cfg.initialization == "random":
        
        print("Starting training with random initialization...")
        
        trained_rbf_model = train_with_rbf(cfg, train_dataloader, test_dataloader, seed, encoder_output=None, save_mode = save_mode, load_model=load_model)
        reconstruction_errors_point_tr, reconstruction_errors_point_te, true_labels_point = Reconstruction_Errors_Clculation(trained_rbf_model, train_dataloader, test_dataloader, cfg)
        Rec_point_wise_results = evaluate_based_on_reconstruction(reconstruction_errors_point_tr, reconstruction_errors_point_te, true_labels_point, cfg)
        SimRec_point_wise_results, SimRec_Addition_point_wise_results = evaluate_based_on_SimRec(trained_rbf_model, train_dataloader, test_dataloader, reconstruction_errors_point_tr, reconstruction_errors_point_te, true_labels_point, cfg)            
        
        # Save the results for this seed
        results[seed] = {
            'Random SimRec Results': SimRec_point_wise_results,
            'Random Rec Results Results': Rec_point_wise_results,
            'Random SimRec Addition Results': SimRec_Addition_point_wise_results
            
        }
        
        print("Training with random initialization completed.")
        
        
    else:
        raise ValueError("Invalid initialization method.")
    

    print(f"Experiment with seed {seed} completed.")

    return results