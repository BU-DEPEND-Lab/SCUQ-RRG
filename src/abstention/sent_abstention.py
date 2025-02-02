import pandas as pd
import numpy as np
import pickle
import random
import argparse


def remove_set_sentences(samples, indices, u_sent_report):
    updated_samples = []
    for sample, uncertainties in zip(samples, u_sent_report):
        sentences = sample.split('. ')
        filtered_sentences = [sentences[j] for j in range(len(sentences)) if j not in indices]
        # Ensure at least one sentence remains in the report
        if len(filtered_sentences) == 0:
            # Find the sentence with the lowest uncertainty to keep
            min_uncertainty_index = np.argmin(uncertainties)
            filtered_sentences.append(sentences[min_uncertainty_index])  # Keep the sentence with the lowest uncertainty
        updated_sample = '. '.join(filtered_sentences)
        updated_samples.append(updated_sample)
    return updated_samples


def gen_sent_pruned_report(pred, u_sent, percentile, max_sent_to_remove, output_path):
    flat_list = [(i, j, item) for i, sublist in enumerate(u_sent) for j, item in enumerate(sublist)]

    cleaned_list = [(i, j, x) for i, j, x in flat_list if not np.isnan(x)]
    cleaned_list.sort(key=lambda x: (x[2], len(u_sent[x[0]])), reverse=True) # uq as primary key and report length as second key

    # number of sentences to remove
    total = len(cleaned_list)
    num_to_remove = int(total * (percentile / 100.0))

    # sentences to remove based on the percentile
    selected_indices = []
    count = 0
    for i, j, value in cleaned_list:
        if count >= num_to_remove:
            break
        selected_indices.append((i, j))
        count += 1

    with open(output_path, "w") as f:
        for index in selected_indices:
            f.write(f"{index}\n")

    print(f"Selected {len(selected_indices)} sentences for removal. {len(selected_indices)/len(cleaned_list)}")
    print('selected',selected_indices[0:5])
    # Remove risky sentences, limiting the removal to `max_sent_to_remove` per report
    purify_report = []
    num_process = len(pred)
    for i in range(num_process):
        if i % 1000 == 0:
            print(f"Processing report {i}")
        report_sentences = [pred[i].split('. ')]  # Split into sentences
        indices_to_remove = [j for j in range(len(u_sent[i])) if (i, j) in selected_indices]

        if len(indices_to_remove) > max_sent_to_remove:
            indices_to_remove = indices_to_remove[:max_sent_to_remove]
            print(indices_to_remove)
            print(f'{i}-th case1')

        tmp = remove_set_sentences([pred[i]], indices_to_remove, [u_sent[i]])
        purify_report.extend(tmp)

    return purify_report


def remove_random_sentences_across_reports(samples, percentile):
    """
    Randomly removes a specified total number of sentences across all reports.

    Parameters:
    - samples: List of reports where each report is a single string of sentences.
    - total_sentences_to_remove: Total number of sentences to remove across all reports.

    Returns:
    - updated_samples: List of reports with sentences removed randomly.
    """
    # Flatten all sentences and collect sentence indices
    sentence_indices = []
    sentence_splits = []
    
    for i, report in enumerate(samples):
        sentences = report.split('. ')
        sentence_splits.append(sentences)  # Store sentences split by report
        sentence_indices.extend([(i, j) for j in range(len(sentences))])  # Store sentence indices per report

    # number of sentences to remove
    total = len(sentence_indices)
    num_to_remove = int(total * (percentile / 100.0))

    indices_to_remove = set(random.sample(sentence_indices, num_to_remove))
    # Generate pruned reports by excluding the selected sentences
    updated_samples = []
    for i, sentences in enumerate(sentence_splits):
        filtered_sentences = [sentence for j, sentence in enumerate(sentences) if (i, j) not in indices_to_remove]
        
        # Ensure at least one sentence remains
        if len(filtered_sentences) == 0:
            min_index = random.choice([j for j in range(len(sentences))])
            filtered_sentences.append(sentences[min_index])

        # Join the pruned sentences into a single string
        updated_samples.append(". ".join(filtered_sentences) + ".")
    return updated_samples


def benchmark_format(pred, path):
    df = pd.DataFrame(pred, columns=['report'])
    df['study_id'] = list(range(0, len(pred)))
    df.to_csv(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sentence Pruning Script')
    parser.add_argument('--pred_path', type=str, default='data/predictions_checkpoints_vicuna-7b-img-report_checkpoint-11200.csv', help='Path to the prediction CSV file')
    parser.add_argument('--oracle_path', type=str, default='data/all_true_report.pkls', help='Path to the oracle pickle file')
    parser.add_argument('--uncertainty_path', type=str, default='data/new_rad_cent_consistent-3858.pkl', help='Path to the sentence-based uncertainty pickle file')
    parser.add_argument('--output_dir', type=str, default='results/exp_result/sent_remove', help='Directory to save output files')
    parser.add_argument("--exp", type=str, default="RaDialog", help="Experiment name")

    args = parser.parse_args()

    # load prediction
    all_pred = pd.read_csv(args.pred_path, header=None)
    all_pred_list = all_pred.values.tolist()
    pred = [item[0] for item in all_pred_list] 

    # load oracle
    with open(args.oracle_path, 'rb') as file:
        all_true_list = pickle.load(file)

    # load sent-based uncertainty
    with open(args.uncertainty_path, 'rb') as file:
        rad_all_cent = pickle.load(file)

    # aggregate uncertainty over 
    u_report = [] # report level uncertainty
    u_sent = [] # sentence level uncertainty 
    for i in range(len(rad_all_cent)):
        tmp = [np.mean(item) for item in rad_all_cent[i]]
        u_report.append(1 - np.mean(tmp))
        u_sent.append([1 - np.mean(sent_10) for sent_10 in rad_all_cent[i]])

    # sentence abstention
    for percent in [3.0, 5.0, 7.0, 9.0, 11.0, 13.0]:
        max_sent_to_remove = 6  
        pruned_report = gen_sent_pruned_report(pred, filtered_u_sent, percent, max_sent_to_remove,
                                               output_path=f'{args.output_dir}/{args.exp}_idx_rad_sent_removed_{percent}.txt')

        benchmark_format(pruned_report, path=f'{args.output_dir}/{args.exp}_rad_sent_removed_{percent}.csv')

        # Random sentence removal based on num_to_remove
        pruned_report_random = remove_random_sentences_across_reports(pred, percent)
        benchmark_format(pruned_report_random, path=f'{args.output_dir}/{args.exp}_random_rad_sent_removed_{percent}.csv')
