### Description
This repository contains files associated with a Master thesis "Extracting Relative Temporal Relation Labels of ICF Functioning Statuses in Dutch Medical Notes" in fullfilment of an MA degree in Linguistics: Text Mining. This thesis in part of an ongoing [A-PROOF project](https://cltl.github.io/a-proof-project/).

This thesis focusses on the development of a classifier for simple relative time extraction of WHO ICF categories out of medical notes by comparing two traditional Machine Learning approaches: feature-explicit SVM model and fine-tuned [medical domain RoBERTa model](https://huggingface.co/CLTL/MedRoBERTa.nl). 
Due to sensitivity of the data, original dataset and model files cannot be shared in this repository.

### Structure of the repository:
```
Thesis-project
└───confusion (confusion matrix visualisations for the final classifiers)
└───corpus_dataset (corpus statistics of the dataset)
└───corpus_error_samples (corpus statistics of the misclassification subsets)
└───descriptives (descriptive statistics of the dataset)
└───irrelevant (scripts used in development of final code)
└───reports (classification reports of the experiments)
└───reports_additional_experiments (classification reports of the excluded experiments)
|   finetuning.py (functions used for fine-tuning the RoBERTa model)
│   LICENSE
|   main.py
|   Meruyert_Nurberdikhanova_Extracting_Relative_Time_MA_Thesis.pdf
│   README.md
│   requirements.txt
|   svm_exp.py (functions used for creating the SVM classifier)
|   utils.py (all additional functions used in code)
```