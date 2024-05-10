#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from Utils import get_data, generate_loaders, set_seed, load_config
from solver import run_experiment
import pickle
import os

##############################

def main(cfg, save_mode, load_model):
    
    # Set the seed for all relevant libraries
    set_seed(42)
    
    #Prepare Dataset    
    (x_train, _), (x_test, y_test_point) = get_data(cfg)
    train_dataloader, test_dataloader = generate_loaders(x_train, x_test, y_test_point, cfg)

    ##############################
    ##############################
    ##############################
    
    seeds = [0]  # List of seeds to use for the experiments
    results = {}  # Dictionary to save the results for each seed
            
    for seed in seeds:
        cfg.seed = seed
        seed_results = run_experiment(cfg, train_dataloader, test_dataloader, seed, save_mode = save_mode, load_model=load_model)
        results[seed] = seed_results

    return results


##############################
if __name__ == "__main__":

    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Run experiments with options to load models and save results.")
    parser.add_argument('--load_model', type=bool, default=False, help='Flag to load a pre-trained model (default: False)')
    parser.add_argument('--save_mode', type=bool, default=False, help='Flag to save the trained model (default: False)')
    
    # Parse arguments
    args = parser.parse_args()

    #Load Config
    working_dir = os.getcwd()   
    cfg = load_config(working_dir)
    
    mode_type = cfg.mode
    selected_model = cfg.model.type
     
    results = main(cfg, save_mode = args.save_mode, load_model = args.load_model )
    
    # Save the results
    with open(f'{cfg.dataset.save_prefix}/results_{cfg.dataset.name}_{selected_model}_mode_{mode_type}_{cfg.initialization}.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {cfg.dataset.save_prefix}")
