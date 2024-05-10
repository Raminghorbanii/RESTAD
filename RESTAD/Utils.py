#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import pairwise_distances
from sklearn.utils import shuffle

import os
import pickle

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, precision_recall_fscore_support, accuracy_score
from sklearn.mixture import GaussianMixture

from sklearn.cluster import KMeans
import random

from hydra import compose, initialize_config_dir
import hashlib


def hash_model_parameters(model):
    """
    Compute a hash of the model's parameters.

    Parameters:
    - model (torch.nn.Module): The PyTorch model.

    Returns:
    - str: The hash value of the model's parameters.
    """
    # Convert model parameters to a byte string
    model_params = []
    for param in model.parameters():
        model_params.append(param.data.cpu().numpy().tobytes())
    model_byte_string = b''.join(model_params)
    
    # Compute the hash of the byte string
    return hashlib.sha256(model_byte_string).hexdigest()



def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")



def load_config(working_dir, config_file_name="base_config"):
    with initialize_config_dir(version_base=None, config_dir=f"{working_dir}/configs"):
        cfg = compose(config_name = config_file_name)
        if cfg.device == "auto":
            cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print('Config File Loaded Dataset:', cfg.dataset.name)
        print('Config File Loaded Model:',cfg.model.type)
        print('Using Device:',cfg.device)

        # If using CUDA, print the GPU model
        if cfg.device == "cuda" and torch.cuda.is_available():
            print('GPU Model:', torch.cuda.get_device_name(0))
            
        return cfg
    

def get_data(cfg, do_normalization=True):
    """
    Get data from npy files

    Return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
    """
    
    prefix = cfg.dataset.data_prefix
    dataset = cfg.dataset.name
    max_train_size = cfg.dataset.max_train_size
    max_test_size = cfg.dataset.max_test_size
    train_start = cfg.dataset.train_start
    test_start = cfg.dataset.test_start
    x_dim = cfg.dataset.x_dim
    
    if max_train_size is None:
        train_end = None
    else:
        train_end = train_start + max_train_size
    if max_test_size is None:
        test_end = None
    else:
        test_end = test_start + max_test_size
        
    print('load data of:', dataset)
    print("train Start and End: ", train_start, train_end)
    print("test Start and End: ", test_start, test_end)
    
    # Load and reshape train data
    train_data = np.load(os.path.join(prefix, dataset + '_train.npy')).reshape((-1, x_dim))
    train_data = train_data[train_start:train_end, :]

    try:
        # Load and reshape test data
        test_data = np.load(os.path.join(prefix, dataset + '_test.npy')).reshape((-1, x_dim))
        test_data = test_data[test_start:test_end, :]
    except FileNotFoundError:
        test_data = None

    try:
        # Load and reshape test labels
        test_label = np.load(os.path.join(prefix, dataset + "_test_label.npy")).reshape((-1))
        test_label = test_label[test_start:test_end]
    except FileNotFoundError:
        test_label = None

    if do_normalization:
        scaler = StandardScaler()  # Fit on training data
        train_data = scaler.fit_transform(train_data)  # Transform training data
        if test_data is not None:
            test_data = scaler.transform(test_data)  # Transform test data based on training data
        
    print("train set shape: ", train_data.shape)
    print("test set shape: ", test_data.shape if test_data is not None else None)
    print("test set label shape: ", test_label.shape if test_label is not None else None)
    
    return (train_data, None), (test_data, test_label)


# Function to create windows from the data
def create_windows(data, window_size, step=1):
    windows = []
    for i in range(0, len(data) - window_size + 1, step):
        windows.append(data[i:i + window_size])
    return np.array(windows)


        
def generate_loaders(train_data, test_data, test_labels, cfg, seed=42):
    """
    Generate DataLoader objects for training and testing datasets.
    
    Parameters:
    - train_data: Training dataset.
    - test_data: Testing dataset.
    - test_labels: Labels for the test dataset.
    - cfg: Configuration object containing batch size, window size, and other parameters.
    - seed: Random seed for reproducibility.
    
    Returns:
    - train_dataloader: DataLoader object for the training dataset.
    - test_dataloader: DataLoader object for the test dataset.
    """
    
    # Extract relevant parameters from configuration object
    batch_size = cfg.model.batch_size
    window_size = cfg.dataset.window_size
    step = cfg.dataset.step
    
    # Segment the data into overlapping windows
    train_data = create_windows(train_data, window_size, step)
    train_data = shuffle(train_data, random_state=seed)

    # Create dummy labels for training data to match its shape
    dummy_train_labels_point = np.zeros_like(train_data, dtype=int)

    test_data = create_windows(test_data, window_size, step)
    test_labels_point = create_windows(test_labels, window_size, step)
    
    
    # Convert data and labels into PyTorch tensors
    train_data = torch.tensor(train_data, dtype=torch.float32)
    test_data = torch.tensor(test_data, dtype=torch.float32)
    test_labels_point = torch.tensor(test_labels_point, dtype=torch.long)

    # Print the shapes of the data tensors (useful for debugging and understanding data dimensions)
    print("train window shape: ", train_data.shape)
    print("test window shape: ", test_data.shape)
    print("test window label shape (point-level): ", test_labels_point.shape)

    # Prepare the training data using a TensorDataset (combining data and labels)
    train_data = TensorDataset(train_data, torch.tensor(dummy_train_labels_point, dtype=torch.long))
    # Create DataLoader objects for both training and testing data
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    
    # For testing data, shuffling isn't needed, so we just specify the batch size
    test_dataset = TensorDataset(test_data, test_labels_point)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader


##############################################
##############################################
##############################################

def get_encoder_output(model, dataloader, device):
    """
    Extracts the encoder outputs from a given model for all batches in a dataloader.

    This function is designed to operate in evaluation mode, meaning that it does not track gradients.
    It's particularly useful for gathering the outputs of a model's encoder component across a dataset,
    which can be useful for tasks like further processing outside
    of the model's training routine (like calculating the mean of the data for RBF layer calculations).

    Parameters:
    - model: The model from which to extract encoder outputs.
    - dataloader: An iterable DataLoader object that provides batches of data to be processed by the model.
                  Each batch should be a tuple where the first element contains the input data.
    - device: The device (CPU, GPU) on which the computations will be performed. This is necessary to ensure
              that data is moved to the correct device before processing.

    Returns:
    - A numpy array containing the concatenated encoder outputs for all batches in the dataloader.
    """

    # Set the model to evaluation mode to disable dropout, batch normalization effects during inference
    model.eval()
    encoder_outputs = []

    # Disable gradient computation for efficiency and to reduce memory usage
    with torch.no_grad():
        for batch in dataloader:
            # Move input data to the specified device
            inputs = batch[0].to(device)

            # Forward pass through the model to get the encoder outputs
            # The model is expected to return a tuple where the encoder output is the second element
            _, enc_out, _ = model(inputs)
            
            # Move the encoder outputs to CPU and convert to numpy for easy manipulation/storage
            encoder_outputs.append(enc_out.cpu().numpy())

    # Concatenate all the encoder outputs along the first dimension to form a single array
    return np.concatenate(encoder_outputs)



def function_data_mean(encoder_output):
    """
    Calculates the mean of the flattened encoder outputs.

    This function takes the encoder outputs, converts them to a PyTorch tensor if they are not already,
    flattens them while preserving the feature dimension, and then calculates the mean across all data points
    in the flattened array. 

    Parameters:
    - encoder_output: A numpy array or a PyTorch tensor containing the encoder outputs. 

    Returns:
    - A PyTorch tensor representing the mean of the flattened encoder outputs across all samples.
    """

    # Convert encoder output to a PyTorch tensor if it's a numpy array
    encoder_output = torch.from_numpy(encoder_output) if not isinstance(encoder_output, torch.Tensor) else encoder_output

    # Flatten the encoder output while preserving the feature dimension (last dimension)
    encoder_output_flat = encoder_output.reshape(-1, encoder_output.size(-1))
    
    # Calculate the mean of the flattened encoder output across the feature dimension
    data_mean = torch.mean(encoder_output_flat, dim=0)

    return data_mean



            
def compute_and_update_centers(encoder_output, rbf_model, cfg):
    """
    Compute centers using K-means on the encoder output and update the RBF layer's centers in the model.
    
    Parameters:
    - encoder_output (torch.Tensor): The output from the encoder.
    - rbf_model (nn.Module): The model containing the RBF layer.
    - cfg (Config): Configuration object with necessary parameters.
    
    Returns:
    - centers (torch.Tensor): The computed centers.
    """
    
    # Reshape the data to treat each time step as a separate sample
    encoder_output = torch.from_numpy(encoder_output)
    encoder_output_flat = encoder_output.reshape(-1, encoder_output.size(-1)).detach().cpu().numpy()
    
    # Apply K-means
    kmeans = KMeans(n_clusters=cfg.model.rbf_dim, init='k-means++', n_init=20, random_state=42).fit(encoder_output_flat)
    
    # Get the centers
    centers = kmeans.cluster_centers_

    # Convert the centers to a PyTorch tensor and move it to the correct device
    centers_tensor = torch.from_numpy(centers).float().to(cfg.device)
    print(f"Centers Shape: {centers_tensor.shape}")

    # Compute pairwise distances between encoder_output and centers
    distances = pairwise_distances(encoder_output_flat, centers)
    
    # Get the minimum squared distance for each data point (i.e., distance to the nearest center)
    min_squared_distances = np.min(distances**2, axis=1)
     
    # Compute the mean of the squared distances
    mean_squared_distance = np.mean(min_squared_distances)
    print(f"Mean Squared Distance Value: {mean_squared_distance}")

    # Ensure mean_squared_distance is not zero to avoid division by zero
    if mean_squared_distance == 0:
        raise ValueError("Mean squared distance is zero. This may cause issues with the initialization of log_gamma.")
        
    
    # Ensure the model has an RBF layer
    if hasattr(rbf_model, 'rbf_layer'):
        # Get initial centers for comparison
        initial_centers = rbf_model.rbf_layer.centers.clone().detach()
        
        # Initialize the RBF layer in the model with these centers
        rbf_model.rbf_layer.centers.data = centers_tensor


        # Before updating log_gamma, store its previous value
        initial_log_gamma = rbf_model.rbf_layer.log_gamma.clone().detach()

        # Update the log_gamma parameter using the mean squared distance
        rbf_model.rbf_layer.log_gamma.data.fill_(torch.log(1.0 / torch.tensor(mean_squared_distance, dtype=torch.float32)))
    
        # Check if centers have been updated
        if torch.equal(initial_centers, rbf_model.rbf_layer.centers):
            print("Centers have not been updated.")
        else:
            print("Centers have been updated.")

        # Check if log_gamma has been updated
        if torch.equal(initial_log_gamma, rbf_model.rbf_layer.log_gamma):
            print("log_gamma has not been updated.")
        else:
            print("log_gamma has been updated.")
    
    else:
        raise ValueError("The provided model does not have an RBF layer.")
    


# Calculate the reconstruction error for each window
def calculate_reconstruction_errors(model, dataloader, cfg, test_mode=False):

    model.eval()
    reconstruction_errors_point = []
    true_labels_point = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(cfg.device)
            outputs, _, _ = model(inputs)
                
            batch_error_point = torch.mean((inputs - outputs) ** 2, dim=2).cpu().numpy() # error per point
            
            reconstruction_errors_point.extend(batch_error_point.flatten())

            if test_mode:
                labels_point = batch[1].numpy()
                true_labels_point.extend(labels_point.flatten())

    if test_mode:
        return np.array(reconstruction_errors_point), np.array(true_labels_point)
    else:
        return np.array(reconstruction_errors_point)



def calculate_rbf_scores(model, dataloader, cfg):
    assert hasattr(model, 'rbf_layer'), "The model does not have an RBF layer."

    model.eval()

    rbf_scores_mean = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(cfg.device)

            _, _, rbf_out = model(inputs)
            
            # RBF score per point by Mean of the scores over centers
            scores_mean = torch.mean(rbf_out, dim=2).cpu().numpy()
            rbf_scores_mean.extend(scores_mean.flatten())

    return np.array(rbf_scores_mean)



def evaluate_rec(reconstruction_errors_tr, reconstruction_errors_te, true_labels, cfg, thresh_type_list=['ratio', 'f1-score'], adjustment_mode_list=[True, False], Analysis_Mode = False): 
    print("##############################")
    results = {}
    
    for thresh_type in thresh_type_list:
        for adjustment_mode in adjustment_mode_list:
            
            print(f"## Calculating evaluate_rec for thresh_type: {thresh_type}, adjustment_mode: {adjustment_mode}")

            if thresh_type == 'ratio':
                #Find threshold based on ratio
                thresh = find_threshold(reconstruction_errors_tr, reconstruction_errors_te, cfg.dataset.anormly_ratio)
        
            elif thresh_type == 'f1-score':
                #Find threshold based on F1-Score
                thresh = calculate_best_f1_threshold(true_labels, reconstruction_errors_te, max_range = 100)

            #Calculate Metrics
            accuracy, precision, recall, f1 = calculate_metrics(true_labels, reconstruction_errors_te, adj_thresh = thresh , adjustment = adjustment_mode)    
        
            # Calculate AUC
            auc = compute_auc(true_labels, reconstruction_errors_te,  adj_thresh = thresh , adjustment = adjustment_mode)
        
            # Calculate AUPRC
            auprc = compute_auprc(true_labels, reconstruction_errors_te,  adj_thresh = thresh , adjustment = adjustment_mode)
            
            results[(thresh_type, adjustment_mode)] = {'accuracy':accuracy, 'precision': precision, 'recall': recall, 'f1-score':f1, 'AUC': auc, 'AUC-PR': auprc}
    

    if Analysis_Mode:
        return thresh, reconstruction_errors_te
    
    else:
        return results




def evaluate_RBFrec(rec_errors_tr, rec_errors_te, rbf_score_tr, rbf_score_te, true_labels, cfg, thresh_type_list=['ratio', 'f1-score'], adjustment_mode_list=[True, False], Analysis_Mode = False):
    print("##############################")
    results = {}
    
    # Normalize the reconstruction errors and RBF scores to the range [0, 1]
    scaler = MinMaxScaler() # fit on training data
    rec_errors_tr = scaler.fit_transform(rec_errors_tr.reshape(-1,1))  # transform training data
    rec_errors_te = scaler.transform(rec_errors_te.reshape(-1,1))  # transform test data
    
    rbf_score_tr = scaler.fit_transform(rbf_score_tr.reshape(-1,1))  # transform training data
    rbf_score_te = scaler.transform(rbf_score_te.reshape(-1,1))  # transform test data

    
    for thresh_type in thresh_type_list:
        for adjustment_mode in adjustment_mode_list:
            
            print(f"## Calculating evaluate_RBFrec for thresh_type: {thresh_type}, adjustment_mode: {adjustment_mode}")

            # Add a small constant to the denominator to avoid division by zero
            anomaly_scores_tr = rec_errors_tr * (1 - rbf_score_tr)
            anomaly_scores_te = rec_errors_te * (1 - rbf_score_te)

            if thresh_type == 'ratio':
                #Find threshold based on ratio
                thresh = find_threshold(anomaly_scores_tr, anomaly_scores_te, cfg.dataset.anormly_ratio)
        
            elif thresh_type == 'f1-score':
                #Find threshold based on F1-Score
                thresh = calculate_best_f1_threshold(true_labels, anomaly_scores_te, max_range = 100)

            #Calculate Metrics
            accuracy, precision, recall, f1 = calculate_metrics(true_labels, anomaly_scores_te, adj_thresh = thresh , adjustment = adjustment_mode)    
        
            # Calculate AUC
            auc = compute_auc(true_labels, anomaly_scores_te,  adj_thresh = thresh , adjustment = adjustment_mode)
        
            # Calculate AUPRC
            auprc = compute_auprc(true_labels, anomaly_scores_te,  adj_thresh = thresh , adjustment = adjustment_mode)
            
            results[(thresh_type, adjustment_mode)] = {'RBFrec accuracy':accuracy, 'RBFrec precision': precision, 'RBFrec recall': recall, 'RBFrec f1-score':f1, 'RBFrec AUC': auc, 'RBFrec AUC-PR': auprc}
    

    if Analysis_Mode:
        return thresh, anomaly_scores_te, rec_errors_te
    
    else:
        return results



def evaluate_RBFrec_Addition(rec_errors_tr, rec_errors_te, rbf_score_tr, rbf_score_te, true_labels, cfg, thresh_type_list=['ratio', 'f1-score'], adjustment_mode_list=[True, False], Analysis_Mode = False):
    print("##############################")
    results = {}

    # Normalize the reconstruction errors and RBF scores to the range [0, 1]
    scaler = MinMaxScaler() # fit on training data
    rec_errors_tr = scaler.fit_transform(rec_errors_tr.reshape(-1,1))  # transform training data
    rec_errors_te = scaler.transform(rec_errors_te.reshape(-1,1))  # transform test data
    
    rbf_score_tr = scaler.fit_transform(rbf_score_tr.reshape(-1,1))  # transform training data
    rbf_score_te = scaler.transform(rbf_score_te.reshape(-1,1))  # transform test data


    for thresh_type in thresh_type_list:
        for adjustment_mode in adjustment_mode_list:
            
            print(f"## Calculating evaluate_RBFrec_Addition for thresh_type: {thresh_type}, adjustment_mode: {adjustment_mode}")

            # Add a small constant to the denominator to avoid division by zero
            anomaly_scores_tr = rec_errors_tr + (1 - rbf_score_tr)
            anomaly_scores_te = rec_errors_te + (1 - rbf_score_te)

            if thresh_type == 'ratio':
                #Find threshold based on ratio
                thresh = find_threshold(anomaly_scores_tr, anomaly_scores_te, cfg.dataset.anormly_ratio)

            elif thresh_type == 'f1-score':
                #Find threshold based on F1-Score
                thresh = calculate_best_f1_threshold(true_labels, anomaly_scores_te, max_range = 100)
                    
            #Calculate Metrics
            accuracy, precision, recall, f1 = calculate_metrics(true_labels, anomaly_scores_te, adj_thresh = thresh , adjustment = adjustment_mode)    
        
            # Calculate AUC
            auc = compute_auc(true_labels, anomaly_scores_te,  adj_thresh = thresh , adjustment = adjustment_mode)
        
            # Calculate AUPRC
            auprc = compute_auprc(true_labels, anomaly_scores_te,  adj_thresh = thresh , adjustment = adjustment_mode)
            
            results[(thresh_type, adjustment_mode)] = {'RBFrec accuracy':accuracy, 'RBFrec precision': precision, 'RBFrec recall': recall, 'RBFrec f1-score':f1, 'RBFrec AUC': auc, 'RBFrec AUC-PR': auprc}
    

    if Analysis_Mode:
        return thresh, anomaly_scores_te, rec_errors_te
    
    else:
        return results



def calculate_rbf_outputs(model, dataloader, cfg, test_mode=False):
    model.eval()
    rbf_outputs = []
    true_labels_window = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(cfg.device)
            _, _, rbf_out = model(inputs)
            rbf_out = rbf_out.mean(axis=(1, 2)).cpu().numpy()
            rbf_outputs.append(rbf_out)
            
            if test_mode:
                labels_window = batch[2].numpy()
                true_labels_window.extend(labels_window)
                
    rbf_outputs = np.concatenate(rbf_outputs, axis=0)
    
    if test_mode:
        return rbf_outputs, np.array(true_labels_window)
    else:
        return rbf_outputs




def find_threshold(rec_error_train, rec_error_test, anormly_ratio):
    combined_rec = np.concatenate([rec_error_train, rec_error_test], axis=0)

    thresh = np.percentile(combined_rec, 100 - anormly_ratio)
    print("Selected Threshold :{:0.10f}".format( thresh))

    return thresh
    


def detection_adjustment(pred, gt):
    
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    
    pred = np.array(pred)
    gt = np.array(gt)
    
    return pred, gt



def compute_auc(y_true, scores, adj_thresh, adjustment = False):
    
    if adjustment:
        scores = (scores > adj_thresh).astype(int)
        scores, y_true = detection_adjustment(scores, y_true)
            
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc_score = auc(fpr, tpr)
    print('AUC: {:0.4f}'.format( auc_score))
    
    return auc_score 


def compute_auprc(y_true, scores, adj_thresh, adjustment = False):
    
    if adjustment:
        scores = (scores > adj_thresh).astype(int)
        scores, y_true = detection_adjustment(scores, y_true)
           
        
    _, recall, thresholds = precision_recall_curve(y_true, scores)
    auc_pr_score = average_precision_score(y_true, scores)
    print('AUC-PR: {:0.4f}'.format( auc_pr_score))
    return auc_pr_score



def calculate_best_f1_threshold(y_true, scores, max_range = 100):

    # Get precision, recall, and threshold values
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    
    # Calculate F1 scores for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall)
    
    # Get the threshold that gives the maximum F1 score
    best_threshold = thresholds[np.argmax(f1_scores)]
    
    print("Selected Threshold :", best_threshold)
    print("Best F1Score :", np.argmax(f1_scores))
    
    return best_threshold



def calculate_metrics(y_true, scores, adj_thresh, adjustment = False):
    y_pred = (scores > adj_thresh).astype(int)
    # print(f"the threshold that we are getting this calculations: {adj_thresh}")
    if adjustment:
        y_pred, y_true = detection_adjustment(y_pred, y_true)
        
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f_score, support = precision_recall_fscore_support(y_true, y_pred, average='binary')

    print(
        "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))

    return accuracy, precision, recall, f_score




def extract_values_from_results(results, keys_dict):
    """
    Extract values from the results based on the provided keys dictionary.
    """
    values_ratio = {}
    for inner_key, keys in keys_dict.items():
        values_ratio[inner_key] = {key: [] for key in keys}
        for outer_key in results:
            for key in keys:
                value_ratio = results[outer_key][outer_key][inner_key][('ratio', False)][key]
                values_ratio[inner_key][key].append(value_ratio)
    return values_ratio


def compute_mean_and_std(values):
    """
    Compute the mean and standard deviation for the provided values.
    """
    mean_std_results = {}
    for inner_key, inner_values in values.items():
        mean_std_results[inner_key] = {}
        for key, value_list in inner_values.items():
            mean_std_results[inner_key][key] = {
                'mean': round(np.mean(value_list), 2),
                'std': round(np.std(value_list), 2)
            }
    return mean_std_results



def flatten_dict(d, prefix=''):
    flat_dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            flat_dict.update(flatten_dict(v, prefix=f"{prefix}{k}_"))
        else:
            clean_key = prefix + str(k)
            clean_key = clean_key.replace(",", "").replace("(", "").replace(")", "").replace(" ", "_").replace("'", "")
            flat_dict[clean_key] = v
    return flat_dict




