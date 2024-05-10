#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from Utils import calculate_reconstruction_errors, evaluate_rec, calculate_rbf_scores, evaluate_RBFrec, evaluate_RBFrec_Addition



def Reconstruction_Errors_Clculation(model, train_dataloader, test_dataloader, cfg):
    """
    Calculate reconstruction errors for both training and test datasets.
    
    Args:
        model: The trained model for which the reconstruction errors are to be calculated.
        train_dataloader: DataLoader object containing the training dataset.
        test_dataloader: DataLoader object containing the test dataset.
        cfg: Configuration object containing necessary parameters.
        
    Returns:
        tuple: Contains reconstruction errors for training dataset, reconstruction errors for test dataset, and true labels for test dataset.
    """
    
    reconstruction_errors_point_tr = calculate_reconstruction_errors(model, train_dataloader, cfg)
    reconstruction_errors_point_te, true_labels_point = calculate_reconstruction_errors(model, test_dataloader, cfg, test_mode=True)

    return reconstruction_errors_point_tr, reconstruction_errors_point_te, true_labels_point



def evaluate_based_on_reconstruction(reconstruction_errors_point_tr, reconstruction_errors_point_te, true_labels_point, cfg):
    """
    Evaluate the model's performance based on the calculated reconstruction errors.
    
    Args:
        reconstruction_errors_point_tr: Reconstruction errors for the training dataset.
        reconstruction_errors_point_te: Reconstruction errors for the test dataset.
        true_labels_point: True labels for the test dataset.
        cfg: Configuration object containing necessary parameters.
        
    Returns:
        dict: Evaluation results based on point-wise reconstruction errors.
    """
    
    point_wise_results = evaluate_rec(reconstruction_errors_point_tr, reconstruction_errors_point_te, true_labels_point, cfg, thresh_type_list=['ratio'], adjustment_mode_list=[False])
    
    return point_wise_results



def RBF_Score_Clculation(model, train_dataloader, test_dataloader, cfg):
    """
    Calculate the RBF Score for both training and test datasets.
    
    Args:
        model: The trained model for which the RBF scores are to be calculated.
        train_dataloader: DataLoader object containing the training dataset.
        test_dataloader: DataLoader object containing the test dataset.
        cfg: Configuration object containing necessary parameters.
        
    Returns:
        tuple: Contains RBF scores for training dataset and RBF scores for test dataset.
    """
    
    rbf_scores_point_tr = calculate_rbf_scores(model, train_dataloader, cfg)
    rbf_scores_point_te = calculate_rbf_scores(model, test_dataloader, cfg)    
    
    return rbf_scores_point_tr, rbf_scores_point_te


def evaluate_based_on_SimRec(model, train_dataloader, test_dataloader, reconstruction_errors_point_tr, reconstruction_errors_point_te, true_labels_point, cfg):
    """
    Evaluate the model's performance based on the SimRec metric.
    
    Args:
        model: The trained model to be evaluated.
        train_dataloader: DataLoader object containing the training dataset.
        test_dataloader: DataLoader object containing the test dataset.
        reconstruction_errors_point_tr: Reconstruction errors for the training dataset.
        reconstruction_errors_point_te: Reconstruction errors for the test dataset.
        true_labels_point: True labels for the test dataset.
        cfg: Configuration object containing necessary parameters.
        
    Returns:
        dict: Evaluation results based on SimRec metric.
    """
    
    rbf_scores_point_tr, rbf_scores_point_te = RBF_Score_Clculation(model, train_dataloader, test_dataloader, cfg)
    SimRec_point_wise_results = evaluate_RBFrec(reconstruction_errors_point_tr, reconstruction_errors_point_te, rbf_scores_point_tr, rbf_scores_point_te, true_labels_point, cfg, thresh_type_list=['ratio'], adjustment_mode_list=[False])
    SimRec_Addition_point_wise_results = evaluate_RBFrec_Addition(reconstruction_errors_point_tr, reconstruction_errors_point_te, rbf_scores_point_tr, rbf_scores_point_te, true_labels_point, cfg, thresh_type_list=['ratio'], adjustment_mode_list=[False])
    
    return SimRec_point_wise_results, SimRec_Addition_point_wise_results




            
            
            