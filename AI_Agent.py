import pickle
import pandas as pd
import numpy as np
import shap
import google.generativeai as genai
from dotenv import load_dotenv, find_dotenv
import os
import vertexai
from vertexai import agent_engines

# Load the pre-trained model
filename = 'mortality_prediction_model.sav'
model = pickle.load(open(filename, 'rb'))

# Load patients from test dataset
test_df = pd.read_csv('AI_agent_test_sepsis_features.csv')
test_df = test_df.sort_values(by=['icustayid', 'charttime'])

feature_cols = [col for col in test_df.columns if col not in ['icustayid', 'charttime', 'mortality_90d', 'bloc', 'Unnamed: 0']]

cols_where_0_is_ok = ['gender', 're_admission', 'mechvent', 'median_dose_vaso', 'max_dose_vaso',
                      'input_total', 'input_4hourly', 'output_total', 'output_4hourly', 'SOFA', 'SIRS']
cols_where_0_means_missing = [col for col in feature_cols if col not in cols_where_0_is_ok]
test_df[cols_where_0_means_missing] = test_df[cols_where_0_means_missing].replace(0, np.nan)

# Only use Imputation Strategy Number 2: Interpolation since it's better
test_df_imp = test_df.copy()
test_df_imp['charttime'] = pd.to_datetime(test_df_imp['charttime'])
test_df_imp = test_df_imp.groupby('icustayid').apply(
    lambda group: group.set_index('charttime').interpolate(method='time').reset_index()
).reset_index(drop=True)

def extract_features_from_test(test_df, feature_cols):
    feature_list = []

    for group_id, group in test_df.groupby('icustayid'):
        features = {}
        for col in feature_cols:
            features[f'{col}_mean'] = group[col].mean()
            features[f'{col}_std'] = group[col].std()
            features[f'{col}_min'] = group[col].min()
            features[f'{col}_max'] = group[col].max()
            features[f'{col}_last'] = group[col].iloc[-1]

        feature_list.append(features)

    X = pd.DataFrame(feature_list)
    return X

test_X = extract_features_from_test(test_df, feature_cols)
test_X_imp = extract_features_from_test(test_df_imp, feature_cols)
test_y_pred = model.predict(test_X)
test_y_imp_pred = model.predict(test_X_imp)

explainer = shap.TreeExplainer(model)
shap_values = explainer(test_X)
shap_values_imp = explainer(test_X_imp)

# Create a LLM that will explain the results of the mortality prediction
mortality_explainer = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    system_instruction=[
        "You are an AI chatbot that explains the results of another AI model that analyzes EHR records.",
        "You will be given a mortality score (1 means they will die in 90 days, 0 means they will not).",
        "Then, you will be given a list of features from the EHR data that were most important in deciding the mortality.",
        "Explain the mortality score and features to a doctor in natural language. Sound confident."
    ],
)

# Create 2 tools that will be binded to the AI agent
# The first one fetches the patient records using icustayid
# The second one predicts mortality, mortality with imputation, and finds the most important features using SHAP
# It then prints out a natural language response to explain these results, using the mortality_explainer LLM
def get_patient_records(icustayid: int):
    """Get patient records according to their icustayid

    Args:
    icustayid: The ID of a patient's ICU stay i.e. a patient's unique ID

    Returns:
    A dataframe of rows which share the same icustayid
    """
    global test_df
    patient_records = test_df[test_df['icustayid'] == icustayid]
    print(patient_records)
    return patient_records

def predict_patient_mortality(icustayid: int):
    """Predict a patient's predicted mortality within 90 days (i.e. mortality_90d)

    Args:
    icustayid: The ID of a patient's ICU stay i.e. a patient's unique ID

    Returns:
    1 if the patient is going to die in 90 days, 0 otherwise
    """
    global test_df, mortality_explainer
    unique_ids = test_df['icustayid'].unique()
    indices = np.where(unique_ids == icustayid)
    idx = indices[0][0]
    mortality_pred = test_y_pred[idx]
    mortality_pred_imp = test_y_imp_pred[idx]

    # Create feature to shap value dict
    patient_shap_values = shap_values[idx]
    patient_shap_values_imp = shap_values_imp[idx]
    feature_shap_dict = {}
    for i in range(len(test_X.columns)):
        feature_shap_dict[test_X.columns[i]] = patient_shap_values.values[i]

    shap.plots.waterfall(patient_shap_values, max_display=10)
    top_10_shaps = sorted(feature_shap_dict.items(), key=lambda item: abs(item[1]), reverse=True)[:10]
    top_10_features = [key for key, value in top_10_shaps]

    chat = mortality_explainer.start_chat()

    message = "The mortality prediction without imputing was " + str(mortality_pred) + ". "
    message += "The mortality prediction with imputing was " + str(mortality_pred_imp) + ". "
    message += "These were the top 10 features according to SHAP value: "
    for ft in top_10_features:
        message += ft + ", "
    message += "Please explain these results in natural language"
    response = chat.send_message(message).text
    print(response)

    return mortality_pred

load_dotenv(find_dotenv())
# Create the AI agent and bind the tools
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

vertexai.init(api_key=os.getenv('GOOGLE_API_KEY'), project=os.getenv('GOOGLE_PROJ_ID1'))
model = "gemini-2.0-flash"
agent = agent_engines.LangchainAgent(
    model=model,
    tools=[get_patient_records, predict_patient_mortality],
)

query = ""
while True:
    print("Enter your query (type 'exit' to quit):")
    query = input()

    if query == "exit":
        break

    response = agent.query(
        input=query
    )
    print(response)

