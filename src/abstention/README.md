## Running Report Abstention Analysis

To generate the results of report abstention for the RaDialog and ChexpertPlus models, use the commands below. Make sure to specify the correct path for SCUQ-RRG before running the commands.

## RaDialog
```shell
python src/abstention/report_abstention.py --exp RaDialog
```
## ChexpertPlus
```shell
python src/abstention/report_abstention.py --exp ChexpertPlus \
  --green_scores_path 'data/green_scores-chexpert-plus-3858.csv' \
  --green_uncertainty_path 'data/chexpert-plus-green_uncertainty-3858.csv' \
  --u_lexicalsim_path 'data/chexpert-plus_lexicalUQ.csv' \
  --output_base_path 'results'
```