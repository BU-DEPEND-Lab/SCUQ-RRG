import pandas as pd
import numpy as np
import pickle
import argparse
from nltk import word_tokenize
from pycocoevalcap.rouge.rouge import Rouge

def preprocess_text(s):
    """
    Helper function to preprocess text by removing special tokens and tokenizing the text.
    """
    s = s.replace('\n', '')
    s = s.replace('<s>', '')
    s = s.replace('</s>', '')
    s = ' '.join(word_tokenize(s.lower()))
    return s

def main(args):
    # Load sampled reports
    with open(args.sampled_reports_path, 'rb') as file:
        all_sample_list = pickle.load(file)

    # Load predicted reports
    all_pred = pd.read_csv(args.predictions_file_path, header=None)
    all_pred_list = all_pred.values.tolist()

    # Initialize ROUGE evaluator
    rouge_evaluator = Rouge()

    average_rouge_l_scores = []

    # Calculate ROUGE-L scores
    for i in range(len(all_pred_list)):
        generated_report = preprocess_text(all_pred_list[i][0])

        sample_scores = []
        for j in range(10):
            sampled_report = preprocess_text(all_sample_list[i][j])

            gts = {0: [generated_report]}  # Generated report as ground-truth
            res = {0: [sampled_report]}    # Sampled report as generated

            rouge_l_score, _ = rouge_evaluator.compute_score(gts, res)
            sample_scores.append(rouge_l_score)

        average_rouge_l_score = np.mean(sample_scores)
        average_rouge_l_scores.append(average_rouge_l_score)

    # Save results to a CSV file
    df = pd.DataFrame(average_rouge_l_scores, columns=["ROUGE_L_UQ"])
    df.to_csv(args.output_path, index=False)
    print(f"Results saved to {args.output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate ROUGE-L scores for sampled reports.')
    parser.add_argument('--sampled_reports_path', type=str, default='data/sampled_reports_num_beams1.pkl',
                        help='Path to the sampled reports file')
    parser.add_argument('--predictions_file_path', type=str, default='data/predictions_checkpoints_vicuna-7b-img-report_checkpoint-11200.csv',
                        help='Path to the predictions file')
    parser.add_argument('--output_path', type=str, default='results/UQ/lexicalUQ.txt',
                        help='Output path for saving the ROUGE-L scores')

    args = parser.parse_args()
    main(args)
