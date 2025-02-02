import pickle
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr


#abstention
def calculate_accuracy_and_improvement(uncertainty_scores, factual_scores):
    all_acc = []
    sorted_indices = np.argsort(uncertainty_scores)

    for i in range(5, 105, 5):
        num = int(len(uncertainty_scores) * i / 100)
        top_indices = sorted_indices[0:num]
        acc = factual_scores[top_indices].mean()
        all_acc.append(acc)
    
    improvements = [(acc - all_acc[-1]) / all_acc[-1] for acc in all_acc]
    
    return all_acc, improvements

# rce evaluation
def calculate_empirical_rce(uncertainty_values, correctness_values, num_bins=20):
    """
    Calculate the Empirical Rank Calibration Error (RCE) for uncertainty and correctness values.
    
    Parameters:
    - uncertainty_values: numpy array of uncertainty values (ug)
    - correctness_values: numpy array of correctness values (us)
    - num_bins: Number of bins to divide the uncertainty values (default is 20)
    
    Returns:
    - empirical_rce: Calculated Empirical Rank Calibration Error
    """
    # Define quantile-based bins
    quantiles = np.linspace(0, 1, num_bins + 1)
    bin_edges = np.quantile(uncertainty_values, quantiles)
    
    # Assign each uncertainty value to a bin
    bin_indices = np.digitize(uncertainty_values, bin_edges, right=True) - 1  # Bin indices for each uncertainty value

    # Initialize lists to store the expected correctness and average uncertainty per bin
    expected_correctness = np.zeros(num_bins)
    average_uncertainty = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)

    # Calculate expected correctness and average uncertainty for each bin
    for i in range(len(uncertainty_values)):
        bin_idx = bin_indices[i]
        if 0 <= bin_idx < num_bins:
            expected_correctness[bin_idx] += correctness_values[i]
            average_uncertainty[bin_idx] += uncertainty_values[i]
            bin_counts[bin_idx] += 1

    # Normalize to get the mean values per bin
    expected_correctness /= np.maximum(bin_counts, 1)
    average_uncertainty /= np.maximum(bin_counts, 1)

    return expected_correctness,average_uncertainty
