'''

Restoration of papers related to the BCI channels.

"Optimal Channel Selection Using Correlation Coefficient for CSP Based EEG Classification"

by jinwoo Lee

'''

import scipy.io as scio

# load data
train_data = scio.loadmat("C:/Users/dsa93/Desktop/compare_paper_other_algorithm/park_optimal_cs_journal/data100Hz/ay/train/train_10")
train_label = scio.loadmat("C:/Users/dsa93/Desktop/compare_paper_other_algorithm/park_optimal_cs_journal/data100Hz/ay/train/train_label_10")
test_data = scio.loadmat("C:/Users/dsa93/Desktop/compare_paper_other_algorithm/park_optimal_cs_journal/data100Hz/ay/test/test_10")
test_label = scio.loadmat("C:/Users/dsa93/Desktop/compare_paper_other_algorithm/park_optimal_cs_journal/data100Hz/ay/test/test_label_10")


# data preprocessing


