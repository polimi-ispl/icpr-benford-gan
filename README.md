# On the use of Benford's law to detect GAN-generated images
This is the official implementation of the paper **On the use of Benford's law to detect GAN-generated images**,
accepted to [ICPR2020](http://www.icpr2020.it/) and currently available on [arXiv](https://arxiv.org/abs/2004.07682).

This repository is currently under maintenance, if you are experiencing any problems, please open an
[issue](https://github.com/polimi-ispl/icpr-benford-gan/issues).

# Getting started

## Prerequisites
1) Install [conda](https://docs.conda.io/en/latest/miniconda.html)
2) Create the `benford-gan` environment with *environment.yml*:
    
    ```bash
    conda env create -f environment.yml
    conda activate benford-gan
    ```
3) Install `pyjpeg-dct` package [1]:
    ```bash
    cd pyjpeg-dct
    pip install -e .
    ```
   If you fail in installing `pyjpeg-dct`, you probably miss the `python3.6-dev` package for your OS. Google is your friend :)

## The whole pipeline

Besides the feature extraction part, the rest of the pipeline is really dependent on the dataset used in the paper. 
Keep this in mind if you try to replicate the paper results. If you just want the feature extraction functions go
straight to [extract_first_digit_hist.py](extract_first_digit_hist.py) and
[extract_features_from_hist.py](extract_features_from_hist.py).

### Feature extraction
1) Insert dataset root (folder containing images) into [params.py](params.py). You probably also need to modify/delete
   lines 5-11 according to the dataset you want to use and its location on your machine. 
2) Extract first digit histograms [extract_first_digit_hist.py](extract_first_digit_hist.py). 
3) Compute divergence features from histograms [extract_features_from_hist.py](extract_features_from_hist.py)
   
### Train
4) Run Random Forest classifier* [rf_combinations_logo.py](rf_combinations_logo.py)
   
### Test
5) Test* [rf_combinations_logo_test_only.py](rf_combinations_logo_test_only.py)

\* NB: to obtain results for `test_compression=False` run  [rf_combinations_logo.py](rf_combinations_logo.py),
to obtain results for `test_compression=True` run  [rf_combinations_logo_test_only.py](rf_combinations_logo_test_only.py)

## SOTA replication
1) Build db [cnn_build_db.py](cnn_build_db.py)
2) Finetune Xception [cnn_finetuning.py](cnn_finetuning.py)
3) Extract cooccurrences [extract_cooccurrences.py](extract_cooccurrences.py)
4) Train SVM on cooccurrences [svm_cooccurrences.py](svm_cooccurrences.py)
5) Train RF on cooccurrences [rf_cooccurrences.py](rf_cooccurrences.py)


## Credits
[Image and Sound Processing Lab - Politecnico di Milano](http://ispl.deib.polimi.it/)
- Nicolò Bonettini
- Paolo Bestagini  
- Simone Milani
- Stefano Tubaro

--- 
[1] Courtesy of [https://github.com/wartmanm/pyjpeg-dct]()