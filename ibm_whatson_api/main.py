#!/usr/bin/env python3

import json
import os
import pandas as pd

from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, ConceptsOptions


# ------------------------------------------ Configuration and Input files ------------------------------------------- #

project_dir = os.path.dirname(os.path.realpath(__file__))
input_dir = os.path.join(os.path.dirname(project_dir), 'input')
output_dir = os.path.join(project_dir, 'out')
commonlitreadabilityprize_input_dir = os.path.join(input_dir, 'commonlitreadabilityprize')

ibm_credentials = {
    "apikey": "aBnPqeUGlfoZS0FAcsKwXNrjpcjUyvIuiXMtvxD_IQhp",
    "iam_apikey_description": "Auto-generated for key 8ba29065-b555-4b14-874a-ec9726b132c7",
    "iam_apikey_name": "Auto-generated service credentials",
    "iam_role_crn": "crn:v1:bluemix:public:iam::::serviceRole:Manager",
    "iam_serviceid_crn": "crn:v1:bluemix:public:iam-identity::a/fe26a52ad72d49baa5769ad6792783aa::serviceid:ServiceId-a8410673-45d8-4e1d-a059-1c2e1b6f73f8",
    "url": "https://api.eu-gb.natural-language-understanding.watson.cloud.ibm.com/instances/f6f5a97d-5653-46f3-a3cb-c81361ae0cd4"
}

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
train_csv.to_csv(os.path.join(output_dir, 'train.csv'))

exit(0)
