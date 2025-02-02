import re
import torch
import torch.nn as nn
import pandas as pd
import pickle
import argparse
import sys
import os
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, root_dir)


from transformers import AutoModelForCausalLM, AutoTokenizer
from green_score.utils import process_responses, make_prompt, tokenize_batch_as_chat, truncate_to_max_len

# A dictionary to store rewards for pairs of reference and hypothesis reports
pair_to_reward_dict = dict()


class GREENModel(nn.Module):
    """
    GREENModel is a neural network model for evaluating radiology reports.

    Args:
        cuda (bool): Whether to use CUDA for GPU acceleration.
        model_id_or_path (str): Path or identifier of the pretrained model.
        do_sample (bool): Whether to sample during generation.
        batch_size (int): Batch size for processing.
        return_0_if_no_green_score (bool): Whether to return 0 if no green score is found.

    Attributes:
        model: Pretrained model for causal language modeling.
        tokenizer: Tokenizer associated with the model.
        categories (list): List of evaluation categories.
        sub_categories (list): List of subcategories for error evaluation.
    """

    def __init__(
            self,
            cuda,
            model_id_or_path,
            do_sample=False,
            batch_size=4,
            return_0_if_no_green_score=True,
    ):
        super().__init__()
        self.cuda = cuda
        self.do_sample = do_sample
        self.batch_size = batch_size
        self.return_0_if_no_green_score = return_0_if_no_green_score
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_id_or_path,
            trust_remote_code=True,
            device_map={"": "cuda:{}".format(torch.cuda.current_device())} if cuda else "cpu",
            torch_dtype=torch.float16,
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_id_or_path,
            add_eos_token=True,
            use_fast=True,
            trust_remote_code=True,
            padding_side="left",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.chat_template = "{% for message in messages %}\n{% if message['from'] == 'human' %}\n{{ '<|user|>\n' + message['value'] + eos_token }}\n{% elif message['from'] == 'system' %}\n{{ '<|system|>\n' + message['value'] + eos_token }}\n{% elif message['from'] == 'gpt' %}\n{{ '<|assistant|>\n'  + message['value'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

        self.categories = [
            "Clinically Significant Errors",
            "Clinically Insignificant Errors",
            "Matched Findings",
        ]

        self.sub_categories = [
            "(a) False report of a finding in the candidate",
            "(b) Missing a finding present in the reference",
            "(c) Misidentification of a finding's anatomic location/position",
            "(d) Misassessment of the severity of a finding",
            "(e) Mentioning a comparison that isn't in the reference",
            "(f) Omitting a comparison detailing a change from a prior study",
        ]

    def get_response(self, input_ids, attention_mask):
        """
        Generates responses using the model and processes them.

        Args:
            input_ids (Tensor): Input IDs for the model.
            attention_mask (Tensor): Attention mask for the input IDs.

        Returns:
            tuple: Processed response list and output IDs.
        """
        outputs = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=self.do_sample,
            max_length=2048,
            temperature=None,
            top_p=None,
        )

        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        response_list = process_responses(responses)

        return response_list, outputs

    def parse_error_counts(self, text, category, for_reward=False):
        """
        Parses error counts from the generated text for a specific category.

        Args:
            text (str): Text to parse for error counts.
            category (str): Category of errors to parse.

        Returns:
            tuple: Sum of counts and list of subcategory counts.
        """
        if category not in self.categories:
            raise ValueError(
                f"Category {category} is not a valid category. Please choose from {self.categories}."
            )

        pattern = rf"\[{category}\]:\s*(.*?)(?:\n\s*\n|\Z)"
        category_text = re.search(pattern, text, re.DOTALL)

        sum_counts = 0
        sub_counts = [0 for i in range(6)]

        if not category_text:
            if self.return_0_if_no_green_score:
                return sum_counts, sub_counts
            else:
                return None, [None for i in range(6)]

        if category_text.group(1).startswith("No"):
            return sum_counts, sub_counts

        if category == "Matched Findings":
            counts = re.findall(r"^\b\d+\b(?=\.)", category_text.group(1))
            if len(counts) > 0:
                sum_counts = int(counts[0])
            return sum_counts, sub_counts

        else:
            sub_categories = [s.split(" ", 1)[0] + " " for s in self.sub_categories]
            matches = sorted(re.findall(r"\([a-f]\) .*", category_text.group(1)))

            if len(matches) == 0:
                matches = sorted(re.findall(r"\([1-6]\) .*", category_text.group(1)))
                sub_categories = [
                    f"({i})" + " " for i in range(1, len(self.sub_categories) + 1)
                ]

            for position, sub_category in enumerate(sub_categories):
                for match in range(len(matches)):
                    if matches[match].startswith(sub_category):
                        count = re.findall(r"(?<=: )\b\d+\b(?=\.)", matches[match])
                        if len(count) > 0:
                            sub_counts[position] = int(count[0])
            return sum(sub_counts), sub_counts

    def compute_green(self, response):
        """
        Computes the green score based on significant clinical errors and matched findings.

        Args:
            response (str): Generated response to evaluate.

        Returns:
            float: Computed green score.
        """
        sig_present, sig_errors = self.parse_error_counts(response, self.categories[0])
        matched_findings, _ = self.parse_error_counts(response, self.categories[2])

        if matched_findings == 0:
            return 0

        if sig_present is None or matched_findings is None:
            return None

        return matched_findings / (matched_findings + sum(sig_errors))

    def forward(self, input_ids, attention_mask):
        """
        Forward pass for the model, computing green scores for input batch.

        Args:
            input_ids (Tensor): Input IDs for the model.
            attention_mask (Tensor): Attention mask for the input IDs.

        Returns:
            tuple: Tensor of green scores and output IDs.
        """
        if self.cuda:
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()

        reward_model_responses, output_ids = self.get_response(input_ids, attention_mask)

        greens = [self.compute_green(response) for response in reward_model_responses]
        greens = [green for green in greens if green is not None]
        return torch.tensor(greens, dtype=torch.float), output_ids


class GREEN(nn.Module):
    """
    GREEN is a wrapper model for GREENModel, handling batching and aggregation.

    Args:
        cuda (bool): Whether to use CUDA for GPU acceleration.

    Attributes:
        model: GREENModel instance for evaluation.
        tokenizer: Tokenizer associated with the model.
    """

    def __init__(self, cuda, max_len=200, **kwargs):
        super().__init__()
        self.cuda = cuda
        self.max_len = max_len
        self.model = GREENModel(cuda, **kwargs)
        self.tokenizer = self.model.tokenizer
        if self.cuda:
            print("Using {} GPUs!".format(torch.cuda.device_count()))
            # self.model = torch.nn.DataParallel(self.model)

    def forward(self, refs, hyps):
        """
        Forward pass for the model, computing green scores for pairs of reference and hypothesis reports.

        Args:
            refs (list): List of reference reports.
            hyps (list): List of hypothesis reports.

        Returns:
            tuple: Mean green score, tensor of green scores, and list of processed responses.
        """
        assert len(refs) == len(hyps)

        refs = truncate_to_max_len(refs, self.max_len)
        hyps = truncate_to_max_len(hyps, self.max_len)

        with torch.no_grad():
            pairs_to_process = []
            final_scores = torch.zeros(len(refs))
            output_ids_dict = {}

            # Iterate over ref-hyp pairs and populate final_scores and pairs_to_process
            for i, (ref, hyp) in enumerate(zip(refs, hyps)):
                if (ref, hyp) in pair_to_reward_dict:
                    final_scores[i], output_ids = pair_to_reward_dict[(ref, hyp)]
                    output_ids_dict[i] = output_ids
                else:
                    pairs_to_process.append((ref, hyp, i))

            if pairs_to_process:
                batch = [make_prompt(ref, hyp) for ref, hyp, _ in pairs_to_process]
                batch = [[{"from": "human", "value": prompt}, {"from": "gpt", "value": ""}] for prompt in batch]
                batch = tokenize_batch_as_chat(self.tokenizer, batch)

                greens_tensor, output_ids = self.model(batch['input_ids'], batch['attention_mask'])

                if len(greens_tensor) == len(pairs_to_process):
                    for (ref, hyp, idx), score, out_id in zip(pairs_to_process, greens_tensor, output_ids):
                        pair_to_reward_dict[(ref, hyp)] = (score, out_id)
                        final_scores[idx] = score
                        output_ids_dict[idx] = out_id
                else:
                    print("An inconsistency was detected in processing pairs.")

            responses = [output_ids_dict[i] for i in range(len(refs))]
            responses = self.tokenizer.batch_decode(responses, skip_special_tokens=True)

            mean_green = final_scores.mean()
            return mean_green, final_scores, process_responses(responses)

def process_batch(model, refs, hyps):
    # Compute GREEN scores
    mean_green, greens, text = model(refs=refs, hyps=hyps)
    return mean_green, greens, text

def main(args):
    model = GREEN(
        model_id_or_path="StanfordAIMI/GREEN-radllama2-7b",
        do_sample=False,  # should be always False
        batch_size=args.batch_size,
        return_0_if_no_green_score=True,
        cuda=True,
    )

    if args.exp_name == 'chexpert-plus':
        with open(args.chexpert_file_path, 'rb') as file:
            all_ = pickle.load(file)
        refs = all_['greedy_reports']
        hyps = [list(item) for item in all_['sampled_reports']]
    else:
        all_pred = pd.read_csv(args.predictions_file_path, header=None)
        all_pred_list = all_pred.values.tolist()

        with open(args.sampled_reports_path, 'rb') as file:
            all_sample_list = pickle.load(file)

        refs = [item[0] for item in all_pred_list]  # greedy decoded as ref
        hyps = all_sample_list  # sampled reports as hyp

    all_greens = []
    all_text = []
    all_green_uncertainty = []

    for i in range(args.num_samples):
        mean_green, greens, text = process_batch(model, [refs[i]] * len(hyps[i]), hyps[i])
        print(f'Green uncertainty for {i}-th sample is {1 - mean_green}')
        all_greens.extend(greens)
        all_text.extend(text)
        all_green_uncertainty.append(1 - mean_green.item())

    # Save results to pickle
    output_path = f'{args.output_base_path}/{args.exp_name}/green_uncertainty-{args.num_samples}.pkl'
    with open(output_path, 'wb') as file:
        pickle.dump({'greens': all_greens, 'text': all_text, 'uncertainty': all_green_uncertainty}, file)
    print(f"Results saved to {output_path}")

    # Save uncertainties to CSV without header
    greens_csv_path = f'{args.output_base_path}/{args.exp_name}/green_uncertainty-{args.num_samples}.csv'
    greens_df = pd.DataFrame(all_green_uncertainty, columns=['Green Uncertainty'])
    greens_df.to_csv(greens_csv_path, index=False, header=False)
    print(f"Greens uncertainty saved to {greens_csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate GREEN scores for a given number of samples.')
    parser.add_argument('--num_samples', type=int, default=3858, help='Number of samples to use for calculation')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for processing samples')
    parser.add_argument('--exp_name', type=str, default='chexpert-plus', help='Experiment name')
    parser.add_argument('--chexpert_file_path', type=str, default='data/batch_chexpert_mimix_cxr_num3858.pkl',
                        help='Path to the CheXpert file')
    parser.add_argument('--sampled_reports_path', type=str, default='data/sampled_reports_num_beams1.pkl',
                        help='Path to the sampled reports file')
    parser.add_argument('--predictions_file_path', type=str, default='data/predictions_checkpoints_vicuna-7b-img-report_checkpoint-11200.csv',
                        help='Path to the predictions file')
    parser.add_argument('--output_base_path', type=str, default='results',
                        help='Base path for output files')
    args = parser.parse_args()

    start_time = time.time()
    main(args)
    print("Total time taken: ", time.time() - start_time)
