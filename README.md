# JSTSP Gan detection based on Benford's law

1) Insert dataset root (folder containing images) into [params.py](params.py).
2) Extract first digit histograms [extract_first_digit_hist.py](extract_first_digit_hist.py)
3) Compute divergence features from histograms [extract_features_from_hist.py](extract_features_from_hist.py)
4) Run Random Forest classifier* [rf_combinations_logo.py](rf_combinations_logo.py)
5) Test* [rf_combinations_logo_test_only.py](rf_combinations_logo_test_only.py)

\* NB: to obtain results for test_compression=False run  [rf_combinations_logo.py](rf_combinations_logo.py),
to obtain results for test_compression=True run  [rf_combinations_logo_test_only.py](rf_combinations_logo_test_only.py)

## SOTA replication (to be checked)
1) Build db [cnn_build_db.py](cnn_build_db.py)
2) Finetune Xception [cnn_finetuning.py](cnn_finetuning.py)
3) Extract cooccurrences [extract_cooccurrences.py](extract_cooccurrences.py)
4) Train SVM on cooccurrences [svm_cooccurrences.py](svm_cooccurrences.py)
5) Train RF on cooccurrences [rf_cooccurrences.py](rf_cooccurrences.py)