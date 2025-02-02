To generate the alignment results of different UQ methods for the RaDialog and ChexpertPlus models, use the following commands. Running these commands will print the Pearson correlation values in a table format. Make sure to specify the correct path for SCUQ-RRG before running the commands.

# Report alignment

## RaDialog
```shell
python3 src/alignment/report_alignment.py --exp RaDialog
```

## ChexpertPlus
```shell
python3 src/alignment/report_alignment.py --exp ChexpertPlus \
  --green_scores_path 'data/green_scores-chexpert-plus-3858.csv' \
  --green_uncertainty_path 'data/chexpert-plus-green_uncertainty-3858.csv' \
  --u_lexicalsim_path 'data/chexpert-plus_lexicalUQ.csv' \
  --report_scores_path 'data/chexpertPlus_report_scores.csv'
```

# Sentence alignment

## RaDialog
```shell
python3 src/alignment/sentence_alignment.py
```


## ChexpertPlus

```shell
python3 src/alignment/sentence_alignment.py --exp CheXpertPlus_mimiccxr \
  --rad_path 'data/CheXpertPlus_mimiccxr_new_rad_cent_consistent-3858.pkl' \
  --sent_precision_path 'data/CheXpertPlus_mimiccxr_sent_rad_precision.pkl'
```