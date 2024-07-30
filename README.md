# WCamNet

Implementation of WCamNet from the paper "Road Surface Friction Estimation for Winter Conditions Utilising General Visual Features"

## Installation

Run:
```
pip3 install -r requirements
```

## Usage

To train and validate, run:
```
python3 train_wcamnet.py -tr <path-to-train-csv> -v <path-to-val-csv> -lr <learning-rate> -wd <weight-decay> -s <path-to-save-directory> -n <name-of-run>
```
To test
```
python3 test_wcamnet.py -w <path-to-weight-file> -te <path-to-test-csv> -s <path-to-save-directory> -n <name-of-run>
```

## .csv data format
The training/validation/testing data should be provided as a .csv-files, which are formatted as
```
<path-to-image>, <friction-value>
```
