import pickle
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import argparse

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def main():
    parser = argparse.ArgumentParser(description="Load and process data for CheXpertPlus or RaDialog experiments.")
    parser.add_argument('--exp', type=str, default='RaDialog', choices=['CheXpertPlus_mimiccxr', 'RaDialog'],
                        help='Experiment type to run: CheXpertPlus_mimiccxr or RaDialog')
    parser.add_argument('--rad_path', type=str, default='data/new_rad_cent_consistent-3858.pkl',
                        help='Path to sentence uncertainty pickle file')
    parser.add_argument('--sent_precision_path', type=str, default='data/RaDialog_sent_rad_precision.pkl',
                        help='Path to sentence level oracle precision pickle file')


    args = parser.parse_args()

    # Load rad data
    rad_all_cent = load_pickle(args.rad_path)
    # Load Sentence Oracle
    sent_true_list = load_pickle(args.sent_precision_path)

    # Aggregate uncertainty over
    u_report = []  # report level uncertainty
    u_sent = []  # sentence level uncertainty
    for i in range(len(rad_all_cent)):
        tmp = [np.mean(item) for item in rad_all_cent[i]]
        u_report.append(1 - np.mean(tmp))
        u_sent.append([1 - np.mean(sent_10) for sent_10 in rad_all_cent[i]])

    all_sent_score_no_nan = [[-1 if np.isnan(item) else item for item in row] for row in sent_true_list]
    u_sent_no_nan = [[1 if np.isnan(item) else item for item in row] for row in u_sent]

    # Flatten lists for correlation calculation
    flattened_u = [item for sublist in all_sent_score_no_nan for item in sublist]
    flattened_s = [item for sublist in u_sent_no_nan for item in sublist]
    flattened_u = np.nan_to_num(flattened_u, nan=0)
    flattened_s = np.nan_to_num(flattened_s, nan=0)
    print(len(flattened_u),len(flattened_s))
    # Calculate Pearson Correlation
    pearson_coeff, _ = pearsonr(flattened_s, flattened_u)
    print(f"Pearson Correlation Coefficient: {pearson_coeff}")



    # Find minimum and maximum indices

    obj = [all_sent_score_no_nan, u_sent_no_nan]  # Alternate between u_sent and all_true_list
    for idx, _ in enumerate(obj):
        current_obj = obj[idx % 2]
        alternate_obj = obj[(idx + 1) % 2]

        min_indices_all = []
        for row in current_obj:
            # minimum value in the row
            min_value = max(row)
            min_indices = [index for index, value in enumerate(row) if value == min_value]
            min_indices_all.append(min_indices)
        max_indices_all = []
        for row in alternate_obj:
            max_value = min(row)
            max_indices = [index for index, value in enumerate(row) if value == max_value]
            max_indices_all.append(max_indices)

    # Calculate mean overlap
        overlap_mean = np.mean([bool(set(min_indices_all[idx]) & set(max_indices_all[idx])) for idx in range(len(min_indices_all))])
        print(f"Sentence alignment acc: {overlap_mean}")


if __name__ == "__main__":
    main()
