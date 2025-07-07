# Home Credit Prediction

## Introduction:
The goal is to create model that predict if client will have payment difficulties or not.

Dataset: [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data)

## Datasets
![image](https://github.com/user-attachments/assets/16d4833d-74a1-4a2f-8a1c-22673905bcb8)

## Project structure:
### other_datasets.ipynb - is a quick look at other dataset provided with other information about clients.
#### For Datasets:
1. Features in each dataset,
2. Duplicated clients,
3. Missing values,
4. Duplicated rows,
5. Distribution of each feature (Numeric and Categorical features).
#### Merging Datasets:
1. Merging bureau and bureau balance by sk_id_bureau
2. Merging the others sk_id_curr
3. Turning to file

### application_train_eda.ipynb - This notebook is EDA and Statistical Inference of application_train data.
1. About Dataset
2. Data Cleaning
3. Exploratory Data Analysis
4. Statistical Inference

### all_models.ipynb - finding best model to use for predicting if client will have payment difficulties or not.
1. Combining and Spliting Data
2. Models with No Feature Engineering
3. Models with Feature Engineering
4. Tuning Models with Feature Engineering

### evaluting_models.ipynb - evaluating all models to pick the best preforming model.
1. Data
2. Evaluation
3. Feature Importance
4. Checking for Overfitting and Underfitting
5. Final Model

## Deployment
* FastAPI app (app.py): Serves predictions at /predict/ from CSV uploads.
* Streamlit app (streamlit_app.py): Simple web UI to upload CSV and show predictions.
* Dockerfile: Containerizes the FastAPI app.
* Model file: catboost_model.pkl with pipeline and cod

### How to Run Locally
1. Start FastAPI: "uvicorn app:app --host 0.0.0.0 --port 8080"
2. In another terminal, run Streamlit: "streamlit run streamlit_app.py"

### How to Deploy on Google Cloud
1. Set project info:
   * "export PROJECT_ID=your-project-id"
   * "export REGION=your-region"

2. Build and push Docker image:
   * "docker build -t gcr.io/$PROJECT_ID/my-fastapi-app ."
   * "docker push gcr.io/$PROJECT_ID/my-fastapi-app"
3. Deploy to Cloud Run: "gcloud run deploy my-fastapi-app --image gcr.io/$PROJECT_ID/my-fastapi-app --platform managed --region $REGION --allow-unauthenticated --port 8080"
4. Update the FastAPI URL in streamlit_app.py to your deployed Cloud Run URL.

Upload a CSV file in the Streamlit app, and it calls the FastAPI endpoint to get predictions.
