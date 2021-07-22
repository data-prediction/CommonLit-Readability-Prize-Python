#!/usr/bin/env python3

import json
import os
import pandas as pd


# ----------------------------------------- Configuration and Output files ------------------------------------------- #

project_dir = os.path.dirname(os.path.dirname(__file__))
output_dir = os.path.join(project_dir, 'out')

with open(os.path.join(output_dir, 'train.csv')) as train_csv_fp:
    train_csv = pd.read_csv(train_csv_fp)


# ------------------------------------------------- Script Execution ------------------------------------------------- #

for train_csv_index, train_csv_row in train_csv.iterrows():
    #  Read IBM response column
    ibm_response = json.loads(train_csv_row.get('ibm_response'))
    print(ibm_response)
