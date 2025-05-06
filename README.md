# AI Agent for Mortality Prediction

This project presents the development of an AI agent designed to predict patient mortality within 90 days based on Electronic Health Records (EHR). The system incorporates data preprocessing, missing value imputation, machine learning models, tool calling, and interpretability techniques to enhance decision-making support in clinical settings. The model primarily leverages Random Forest for classification and SHAP (SHapley Additive exPlanations) for explainability. Furthermore, the predictive results are translated into natural language using a large language model (LLM), offering clinicians an intuitive understanding of patient risk profiles. The final product is an AI agent accessible through natural language queries, implemented using Langchain and Google's Gemini-2.0.

## How to run the experiment used in the paper
1. Open the `AI_Agent_for_EHR_Data.ipynb` locally or in Google Colab
2. Run each cell sequentially

## How to use the AI_Agent.py module


## File Explanation
* `AI_Agent.py` - Contains an AI Agent using a pre-trained model to answer queries on mortality prediction
* `AI_Agent_for_EHR_Data.ipynb` - Runs the experiments that produce the results used in the paper
* `AI_Agent_EHR_Analysis.pdf` - A paper/report explaining the pipeline behind the AI Agent and the experiments that were run to measure its performance

## Dependencies
