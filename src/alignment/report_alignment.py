import pickle
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import argparse
import os

def main(args):
    if args.exp == "RaDialog":
        run_radialog(args)
    elif args.exp == "ChexpertPlus":
        run_chexpertplus(args)
    else:
        raise ValueError("Invalid experiment name. Use 'RaDialog' or 'ChexpertPlus'.")

def run_radialog(args):
    # Load data
    with open(args.green_scores_path, 'rb') as file:
        green_score = pickle.load(file)
    us = np.array([t.numpy() for t in green_score['greens']])
    
    u_nll = pd.read_csv(args.u_nll_path)['u_nll'].values
    u_normnll = pd.read_csv(args.u_normnll_path)['u_normnll'].values
    
    with open(args.green_uncertainty_path, 'rb') as file:
        green_uncertainty = pickle.load(file)
    ugreen = np.array([t.numpy() for t in green_uncertainty['uncertainty']])
    
    u_lexicalsim = pd.read_csv(args.u_lexicalsim_path)['ROUGE_L_UQ'].values
    u_lexicalsim = [1 - i for i in u_lexicalsim]
    
    res = pd.read_csv(args.report_scores_path)
    
    u = [ugreen,u_nll,u_normnll,u_lexicalsim]
    u_name = ['VRO-GREEN','Predictive Entropy','Normalized Entropy','Lexical Similarity']

    for i in range(len(u)):
        # Green-based uncertainty
        tmp = [[hd, pearsonr(u[i][:-1], res[hd])[0]] for hd in res.keys()[4:]]
        tmp.append(['Green', pearsonr(u[i], np.array([t.numpy() for t in green_score['greens']]))[0]])

        print(f"RaDialog Pearson Correlation Results({u_name[i]}):")
        for hd, value in tmp:
            print(f"{hd}: {value}")

def run_chexpertplus(args):
    # Load data
    score = pd.read_csv(args.green_scores_path, header=None)
    score = np.array([float(t.replace("tensor(", "").replace(")", "")) for t in score[0].values])
    
    ugreen = pd.read_csv(args.green_uncertainty_path, header=None)
    ugreen = np.array([float(t.replace("tensor(", "").replace(")", "")) for t in ugreen[0].values])
    
    res = pd.read_csv(args.chexpert_plus_path)
    
    # Green-based uncertainty
    tmp = [[hd, pearsonr(ugreen, res[hd])[0]] for hd in res.keys()[5:]]
    tmp.append(['Green', pearsonr(ugreen, score)[0]])
    print("ChexpertPlus Pearson Correlation Results (Green-based Uncertainty):")
    for hd, value in tmp:
        print(f"{hd}: {value}")
    
    # LexicalSim uncertainty
    u_lexicalsim = pd.read_csv(args.u_lexicalsim_path)['ROUGE_L_UQ'].values
    tmp = [[hd, pearsonr(-u_lexicalsim, res[hd])[0]] for hd in res.keys()[5:]]
    tmp.append(['Green', pearsonr(-u_lexicalsim, score)[0]])
    print("ChexpertPlus Pearson Correlation Results (LexicalSim-based Uncertainty):")
    for hd, value in tmp:
        print(f"{hd}: {value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Pearson Correlation Results for RaDialog and ChexpertPlus Experiments")
    parser.add_argument('--exp', type=str, required=True, help="Experiment name: 'RaDialog' or 'ChexpertPlus'")
    parser.add_argument('--green_scores_path', type=str, default='data/green_scores-3858.pkl', help="Path to the green scores file")
    parser.add_argument('--u_nll_path', type=str, default='data/uq/u_nll.csv', help="Path to the u_nll CSV file (for RaDialog only)")
    parser.add_argument('--u_normnll_path', type=str, default='data/uq/u_normnll.csv', help="Path to the u_normnll CSV file (for RaDialog only)")
    parser.add_argument('--green_uncertainty_path', type=str, default='data/green_uncertainty-3858.pkl', help="Path to the green uncertainty file")
    parser.add_argument('--u_lexicalsim_path', type=str, default='data/uq/lexicalUQ.csv', help="Path to the lexical similarity CSV file")
    parser.add_argument('--report_scores_path', type=str, default='data/report_scores_-1.csv', help="Path to the report scores file")
    parser.add_argument('--chexpert_plus_path', type=str, default='data/chexpertPlus_report_scores.csv', help="Path to the ChexpertPlus batch file (for ChexpertPlus only)")
    args = parser.parse_args()
    main(args)
