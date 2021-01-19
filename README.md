# JSTSP Gan detection based on Benford's law

1) Insert dataset root (folder containing images) into [params.py](params.py).
2) Extract first digit histograms [extract_first_digit_hist.py](extract_first_digit_hist.py)
3) Compute divergence from histograms (compact features) [extract_features_from_hist.py](extract_features_from_hist.py)
4) Run Random Forest classifier* [rf_combinations_logo.py](rf_combinations_logo.py)
5) Test* [rf_combinations_logo_test_only.py](rf_combinations_logo_test_only.py)

\* NB: to obtain results for test_compression=False run  [rf_combinations_logo.py](rf_combinations_logo.py),
to obtain results for test_compression=True run  [rf_combinations_logo_test_only.py](rf_combinations_logo_test_only.py)
## SOTA replication
1) Build db [cnn_build_db.py](cnn_build_db.py)
2) Finetune CNN [cnn_finetuning.py](cnn_finetuning.py)