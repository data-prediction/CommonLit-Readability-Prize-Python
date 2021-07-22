import json
import os
import pandas as pd
import sys

from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, ConceptsOptions


# ------------------------------------------ Configuration and Input files ------------------------------------------- #

project_dir = os.path.dirname(sys.argv[0])
input_dir = os.path.join(os.path.dirname(project_dir), 'input')
out_dir = os.path.join(project_dir, '../out')
commonlitreadabilityprize_input_dir = os.path.join(input_dir, 'commonlitreadabilityprize')

with open(os.path.join(project_dir, 'config', 'ibm_credentials.json')) as ibm_credentials_fp:
    ibm_credentials = json.load(ibm_credentials_fp)

with open(os.path.join(commonlitreadabilityprize_input_dir, 'train.csv')) as train_csv_fp:
    train_csv = pd.read_csv(train_csv_fp)


# ----------------------------------------------- IBM API integration ------------------------------------------------ #

authenticator = IAMAuthenticator(ibm_credentials.get('apikey'))
natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2021-03-25',
    authenticator=authenticator
)

natural_language_understanding.set_service_url(ibm_credentials.get('url'))


# ------------------------------------------------- Script Execution ------------------------------------------------- #

ibm_responses = [None] * len(train_csv)

for train_csv_index, train_csv_row in train_csv.iterrows():
    #  Call IBM API
    ibm_response = natural_language_understanding.analyze(
        text=train_csv_row.get('excerpt'),
        features=Features(concepts=ConceptsOptions())
    ).get_result()
    # Save IBM response
    ibm_responses[train_csv_index] = json.dumps(ibm_response, indent=2)
    break

# Set ibm_response column
train_csv['ibm_response'] = ibm_responses

# Write DataFrame CSV in ".out/" folder
train_csv.to_csv(os.path.join(out_dir, 'train.csv'))

exit(0)
