import json, os, sys, pandas
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, ConceptsOptions


# ------------------------------------------ Configuration and Input files ------------------------------------------- #

project_dir = os.path.dirname(sys.argv[0])
input_dir = os.path.join(os.path.dirname(project_dir), 'input')
commonlitreadabilityprize_input_dir = os.path.join(input_dir, 'commonlitreadabilityprize')

with open(os.path.join(project_dir, 'config', 'ibm_credentials.json')) as ibm_credentials_fp:
    ibm_credentials = json.load(ibm_credentials_fp)

with open(os.path.join(commonlitreadabilityprize_input_dir, 'train.csv')) as train_csv_fp:
    train_csv = pandas.read_csv(train_csv_fp)

print(train_csv)

# ----------------------------------------------- IBM API integration ------------------------------------------------ #

authenticator = IAMAuthenticator(ibm_credentials.get('apikey'))
natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2021-03-25',
    authenticator=authenticator
)

natural_language_understanding.set_service_url(ibm_credentials.get('url'))


exit(0)


#  Call IBM API
response = natural_language_understanding.analyze(
    url='http://newsroom.ibm.com/Guerbet-and-IBM-Watson-Health-Announce-Strategic-Partnership-for-Artificial'
        '-Intelligence-in-Medical-Imaging-Liver',
    features=Features(concepts=ConceptsOptions())
).get_result()

print(json.dumps(response, indent=2))
