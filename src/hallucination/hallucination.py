import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import random

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def count_substrings(pred, reference_substrings):
    substring_count_dict = defaultdict(int)
    for idx, report in enumerate(pred):
        report = str(report)
        report_lower = report.lower()
        count = sum(1 for substring in reference_substrings if substring in report_lower)
        substring_count_dict[idx] = count
    return dict(substring_count_dict)

def plot_distribution(substring_count_dict):
    value_distribution = Counter(substring_count_dict.values())
    plt.figure(figsize=(10, 6))
    plt.bar(value_distribution.keys(), value_distribution.values(), width=0.5, edgecolor='black')
    plt.title('Distribution of Substring Counts in Reports')
    plt.xlabel('Count of Substrings')
    plt.ylabel('Number of Reports')
    plt.show()

def exclude_top_uncertainty_analysis(pred, reference_substrings, ugreen, top_percent=0.2):
    threshold = sorted(ugreen, reverse=True)[int(len(ugreen) * top_percent) - 1]
    remaining_indices = [idx for idx, uncertainty in enumerate(ugreen) if uncertainty < threshold]
    remaining_pred = [pred[idx] for idx in remaining_indices]
    excluded_substring_count_dict = count_substrings(remaining_pred, reference_substrings)
    random_indices = random.sample(range(len(pred)), len(remaining_indices))
    random_pred = [pred[idx] for idx in random_indices]
    random_substring_count_dict = count_substrings(random_pred, reference_substrings)
    excluded_substring_count_dict = {remaining_indices[idx]: count for idx, count in excluded_substring_count_dict.items()}
    random_substring_count_dict = {random_indices[idx]: count for idx, count in random_substring_count_dict.items()}
    return excluded_substring_count_dict, random_substring_count_dict

def main():
    parser = argparse.ArgumentParser(description="Process radiology experiments and generate uncertainty plots.")
    parser.add_argument('--exp', type=str, default='RaDialog', choices=['CheXpertPlus_mimiccxr', 'RaDialog'],
                        help='Experiment type to run: CheXpertPlus_mimiccxr or RaDialog')
    parser.add_argument('--pred_path', type=str, default='data/predictions_checkpoints_vicuna-7b-img-report_checkpoint-11200.csv',
                        help='Path to prediction data')
    parser.add_argument('--uncertainty_path', type=str, default='data/green_uncertainty-3858.pkl',
                        help='Path to uncertainty data')
    parser.add_argument('--output_path', type=str, default='results/',
                        help='Directory to save output plots')
    args = parser.parse_args()

    # Load predictions
    if args.exp == 'CheXpertPlus_mimiccxr':
        with open(args.pred_path, 'rb') as file:
            chexpert_plus = pickle.load(file)
        pred = chexpert_plus['greedy_reports']
    else:
        all_pred = pd.read_csv(args.pred_path,header=None)
        all_pred_list = all_pred.values.tolist()
        pred = [item[0] for item in all_pred_list]

    # Load uncertainty data
    if args.exp == 'CheXpertPlus_mimiccxr':
        ugreen = pd.read_csv(args.uncertainty_path, header=None)
        ugreen = np.array([float(t.replace("tensor(", "").replace(")", "")) for t in ugreen[0].values])
    else:
        with open(args.uncertainty_path, 'rb') as file:
            green_uncertainty = pickle.load(file)
        ugreen = np.array([t.numpy() for t in green_uncertainty['uncertainty']])

    # Define reference substrings
    reference_substrings = {
        "more", "regress", "advance", "less", "fewer", "constant", "unchanged", "prior", 
        "new", "stable", "progressed", "interval", "previous", "further", "again", "since", 
        "increase", "improve", "remain", "worse", "persist", "remov", "similar", "cleared", 
        "earlier", "existing", "decrease", "reduc", "recurrent", "redemonstrat", "resolv", 
        "still", "has", "enlarged", "lower", "larger", "extubated", "smaller", "higher", 
        "continue", "compar", "change", "develop", "before"
    }

    # Generate results and plots
    top_substring_count_dict, random_substring_count_dict = exclude_top_uncertainty_analysis(pred, reference_substrings, ugreen, top_percent=0.2)
    non_zero_count = sum(1 for count in top_substring_count_dict.values() if count > 0)
    print(f'Percentage of reports with prior reference (excluding top uncertainty): {non_zero_count / len(top_substring_count_dict)}')
    print(f'Number of prior substrings per report (excluding top uncertainty): {np.mean(list(top_substring_count_dict.values()))}')

    rd_non_zero_counts = []
    rd_avg_prior_substrings = []
    for _ in range(5):
        _, random_substring_count_dict = exclude_top_uncertainty_analysis(pred, reference_substrings, ugreen, top_percent=0.2)
        rd_non_zero_count = sum(1 for count in random_substring_count_dict.values() if count > 0)
        rd_non_zero_counts.append(rd_non_zero_count / len(random_substring_count_dict))
        rd_avg_prior_substrings.append(np.mean(list(random_substring_count_dict.values())))
    avg_rd_non_zero_count = np.mean(rd_non_zero_counts)
    avg_rd_avg_prior_substrings = np.mean(rd_avg_prior_substrings)
    print(f'Random (average over 5 runs) - Percentage of reports with prior reference: {avg_rd_non_zero_count}')
    print(f'Random (average over 5 runs) - Number of prior substrings per report: {avg_rd_avg_prior_substrings}')

    # Plotting the results
    percent_reports_with_prior = []
    num_prior_substrings_per_report = []
    substring_count_dict = count_substrings(pred, reference_substrings)
    non_zero_count = sum(1 for count in substring_count_dict.values() if count > 0)
    percent_reports_with_prior.append(non_zero_count / len(substring_count_dict))
    num_prior_substrings_per_report.append(np.mean(list(substring_count_dict.values())))

    top_percent_values = np.arange(0, 1.0, 0.1)
    random_percent_reports_with_prior_avg = [percent_reports_with_prior[0]]
    random_num_prior_substrings_per_report_avg = [num_prior_substrings_per_report[0]]

    for top_percent in top_percent_values[1:]:
        top_substring_count_dict, _ = exclude_top_uncertainty_analysis(pred, reference_substrings, ugreen, top_percent=top_percent)
        non_zero_count = sum(1 for count in top_substring_count_dict.values() if count > 0)
        percentage_with_prior = non_zero_count / len(top_substring_count_dict) if len(top_substring_count_dict) > 0 else 0
        avg_prior_substrings = np.mean(list(top_substring_count_dict.values())) if top_substring_count_dict else 0
        percent_reports_with_prior.append(percentage_with_prior)
        num_prior_substrings_per_report.append(avg_prior_substrings)

        random_percent_reports_with_prior = []
        random_num_prior_substrings_per_report = []
        for _ in range(5):
            _, random_substring_count_dict = exclude_top_uncertainty_analysis(pred, reference_substrings, ugreen, top_percent=top_percent)
            rd_non_zero_count = sum(1 for count in random_substring_count_dict.values() if count > 0)
            rd_percentage_with_prior = rd_non_zero_count / len(random_substring_count_dict) if len(random_substring_count_dict) > 0 else 0
            rd_avg_prior_substrings = np.mean(list(random_substring_count_dict.values())) if random_substring_count_dict else 0
            random_percent_reports_with_prior.append(rd_percentage_with_prior)
            random_num_prior_substrings_per_report.append(rd_avg_prior_substrings)
        random_percent_reports_with_prior_avg.append(np.mean(random_percent_reports_with_prior))
        random_num_prior_substrings_per_report_avg.append(np.mean(random_num_prior_substrings_per_report))

    # Plotting the results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), dpi=300)
    top_percent_labels = [f"{int(x * 100)}%" for x in top_percent_values]

    ax1.plot(top_percent_values, [p * 100 for p in percent_reports_with_prior], marker='o', linestyle='-', linewidth=1.5, markersize=6, color='blue', label='VRO-GREEN Guided Abstention')
    ax1.plot(top_percent_values, [p * 100 for p in random_percent_reports_with_prior_avg], marker='x', linestyle='--', linewidth=1.5, markersize=6, color='red', label='Random Baseline (Avg of 5)')
    ax1.set_title('Reports with Prior Reference', fontsize=14, weight='bold')
    ax1.set_xlabel('Reports Abstained (%)', fontsize=12)
    ax1.set_ylabel('Reports with Prior (%)', fontsize=12)
    ax1.set_xticks(top_percent_values)
    ax1.set_xticklabels(top_percent_labels)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax1.grid(True, linestyle='--', linewidth=0.5)
    ax1.legend(fontsize=10, loc='lower left', frameon=True, edgecolor='black')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))

    ax2.plot(top_percent_values, num_prior_substrings_per_report, marker='s', linestyle='-', linewidth=1.5, markersize=6, color='orange', label='VRO-GREEN Guided Abstention')
    ax2.plot(top_percent_values, random_num_prior_substrings_per_report_avg, marker='x', linestyle='--', linewidth=1.5, markersize=6, color='red', label='Random Baseline (Avg of 5)')
    ax2.set_title('Prior Substrings per Report', fontsize=14, weight='bold')
    ax2.set_xlabel('Reports Abstained (%)', fontsize=12)
    ax2.set_ylabel('Prior Substrings (Avg)', fontsize=12)
    ax2.set_xticks(top_percent_values)
    ax2.set_xticklabels(top_percent_labels)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    ax2.grid(True, linestyle='--', linewidth=0.5)
    ax2.legend(fontsize=10, loc='lower left', frameon=True, edgecolor='black')

    plt.tight_layout(pad=3.0)
    plt.savefig(f"{args.output_path}/{args.exp}_figure3.png", format='png')
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()