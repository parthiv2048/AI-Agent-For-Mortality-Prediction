# AI Agent for Mortality Prediction

This project presents the development of an AI agent designed to predict patient mortality within 90 days based on Electronic Health Records (EHR). The system incorporates data preprocessing, missing value imputation, machine learning models, tool calling, and interpretability techniques to enhance decision-making support in clinical settings. The model primarily leverages Random Forest for classification and SHAP (SHapley Additive exPlanations) for explainability. Furthermore, the predictive results are translated into natural language using a large language model (LLM), offering clinicians an intuitive understanding of patient risk profiles. The final product is an AI agent accessible through natural language queries, implemented using Langchain and Google's Gemini-2.0.

## How to run the experiments used in the paper
1. Open the `AI_Agent_for_EHR_Data.ipynb` locally or in Google Colab
2. Run each cell sequentially

## How to use the AI_Agent.py module
1. Make sure you have a .env file in the same folder as `AI_Agent.py`. Your .env file should contain your Google Cloud API Key as `GOOGLE_API_KEY` and you Google Cloud Project ID as `GOOGLE_PROJ_ID1`. On your Google Cloud Console, VertexAI must be enabled.
2. Install all required dependencies by running `pip install -r requirements.txt`
3. Make sure the pre-trained model `mortality_prediction_model.sav` and the testing dataset `AT_agent_test_sepsis_features.csv` are in the same folder as `AI_Agent.py`
4. Run the file using `python AI_Agent.py`
5. The AI Agent currently supports 2 types of queries - fetching patient records using `icustayid` and predicting and explaining mortality for a patient using `icustayid`

## File Explanation
* `AI_Agent.py` - Contains an AI Agent using a pre-trained model to answer queries on mortality prediction
* `AI_Agent_for_EHR_Data.ipynb` - Runs the experiments that produce the results used in the paper
* `AI_Agent_EHR_Analysis.pdf` - A paper/report explaining the pipeline behind the AI Agent and the experiments that were run to measure its performance
* `AI_agent_train_sepsis.csv` - Training dataset for the mortality prediction model
* `AI_agent_test_sepsis_features.csv` - Testing dataset for the mortality prediction model (missing class labels) and also the dataset used by the AI Agent
* `mortality_prediction_model.sav` - Random Forest Classifier model trained on the entire training dataset `AI_agent_train_sepsis.csv` (after missing values were imputed using linear interpolation and SMOTE was used to oversample the minority positive class label)

## Dependencies
pandas==2.2.2

numpy==2.0.2

imblearn==0.13.0

scikit-learn==1.6.1

shap==0.47.2

matplotlib==3.10.0

vertexai==1.91.0

langchain[google-vertexai]
