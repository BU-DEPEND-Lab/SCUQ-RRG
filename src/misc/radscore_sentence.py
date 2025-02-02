import argparse
import pickle
import pandas as pd
from radgraph import F1RadGraph

def count_consistent_entities_batch(hypothesis_annotation_lists_batch, reference_annotation_lists_batch):
    consistent_ratios = []
    for i in range(len(hypothesis_annotation_lists_batch)):
        hypothesis_entities = hypothesis_annotation_lists_batch[i]['entities']
        reference_entities = reference_annotation_lists_batch[i]['entities']
        reference_set = list((entity['tokens'], entity['label']) for entity in reference_entities.values())
        hypothesis_set = list((entity['tokens'], entity['label']) for entity in hypothesis_entities.values())

        consistent_count = 0
        for entity in hypothesis_entities.values():
            if (entity['tokens'], entity['label']) in reference_set:
                consistent_count += 1
        if len(hypothesis_set) > 0:
            ratio = consistent_count / len(hypothesis_set)
        else:
            ratio = float('nan')
        consistent_ratios.append(ratio)
    return consistent_ratios

def produce_scores(args):
    print(args.exp)

    # Load prediction
    if args.exp == 'CheXpertPlus_mimiccxr':
        with open(args.chexpert_file, 'rb') as file:
            chexpert_plus = pickle.load(file)
        pred = chexpert_plus['greedy_reports']
    else:
        all_pred = pd.read_csv(args.predictions_file, header=None)
        all_pred_list = all_pred.values.tolist()
        pred = [item[0] for item in all_pred_list]

    # Load oracle
    with open(args.oracle_file, 'rb') as file:
        all_true_list = pickle.load(file)

    # Load RadGraph
    mode = "complete"
    f1radgraph = F1RadGraph(reward_level=mode)

    all_sent_score = []
    num_samples = len(pred) if args.num_samples is None else args.num_samples
    for i in range(num_samples):
        if i % 10 == 0:
            print(i)
        samples = pred[i]
        sent_score = []
        for sent in samples.split('. '):
            mean_reward, reward_list, hypothesis_annotation_lists, reference_annotation_lists = f1radgraph(hyps=[sent], refs=[all_true_list[i]])
            tmp = count_consistent_entities_batch(hypothesis_annotation_lists, reference_annotation_lists)
            if len(tmp) == 0:
                tmp = [float('nan')]
            sent_score.append(tmp[0])
        all_sent_score.append(sent_score)

    output_file = f"{args.output_dir}/{args.exp}-sent_rad_precision-{args.num_samples}.pkl"
    with open(output_file, 'wb') as file:
        pickle.dump(all_sent_score, file)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Produce some scores.")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to process")
    parser.add_argument("--exp", type=str, default="CheXpertPlus_mimiccxr", help="Experiment name")
    parser.add_argument("--chexpert_file", type=str, default="data/batch_chexpert_mimix_cxr_num3858.pkl",
                        help="Path to the CheXpert file")
    parser.add_argument("--predictions_file", type=str, default="data/predictions_checkpoints_vicuna-7b-img-report_checkpoint-11200.csv",
                        help="Path to the predictions file")
    parser.add_argument("--oracle_file", type=str, default="data/all_true_report.pkls",
                        help="Path to the oracle file")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save the output files")

    args = parser.parse_args()
    produce_scores(args)
