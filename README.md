AI-Powered Wafer Yield Analytics
Project Overview
This project aims to revolutionize semiconductor manufacturing efficiency, particularly for OSAT (Outsourced Semiconductor Assembly and Test) lines in India, by leveraging advanced Artificial Intelligence. The core objective is to predict wafer-level yield losses proactively, enabling manufacturers to intervene swiftly, optimize processes, and significantly reduce costly scrap.

The Problem
In semiconductor manufacturing, low wafer-level yield prediction leads to substantial scrap rates (estimated at 3-5% in Indian OSAT facilities). This results in significant financial losses (potentially ₹50 Cr per fab per year for even a 1% improvement) and impacts the domestic electronics supply chain. Current methods often lack the precision and foresight needed to prevent these losses effectively.

Our Solution
We are developing a cloud-native, AI-powered SaaS platform that transforms raw sensor data into actionable insights. Our solution focuses on:

Predictive Analytics: Utilizing machine learning to forecast potential yield issues.

Root Cause Analysis: Providing interpretability for predictions to identify underlying factors affecting yield.

Process Optimization: Enabling manufacturers to make data-driven decisions to enhance efficiency and quality.

Architecture & Technology Stack
The project follows a microservice-based architecture, leveraging modern data science and web technologies:

Data Storage: Raw and preprocessed wafer sensor data.

data/raw/: Stores raw datasets (e.g., secom.data, secom_labels.data).

data/processed/: Stores cleaned and preprocessed data (e.g., secom_preprocessed.csv).

Data Processing & Modeling:

Python: For data manipulation, machine learning, and interpretability.

Pandas & NumPy: For data handling.

XGBoost: The chosen machine learning model for yield prediction.

Imblearn (SMOTE): Used to address class imbalance during model training.

SHAP: For model interpretability and root cause analysis.

Jupyter Notebooks: (notebooks/data_exploration.ipynb, notebooks/model_training.ipynb) for iterative development, experimentation, and analysis.

Backend (API):

FastAPI: A modern, fast (high-performance) web framework for building the prediction microservice.

src/api/main.py: Contains the FastAPI application and the /score endpoint.

src/model/model_smote.pkl: Stores the trained and pickled XGBoost model.

Frontend (User Interface):

React: For building a dynamic and interactive web application (planned).

Plotly: For data visualization, including wafer heatmaps (planned).

Deployment:

Docker: For containerizing the FastAPI application, ensuring portability and consistent environments.

Azure App Service: Target cloud platform for deployment (planned).

Project Structure
ai_yield_analytics/
├── data/
│   ├── raw/                  # Original, unprocessed datasets
│   └── processed/            # Cleaned and preprocessed data
├── notebooks/
│   ├── data_exploration.ipynb # For initial data analysis and cleaning
│   └── model_training.ipynb   # For model training, evaluation, and saving
├── src/
│   ├── api/                  # FastAPI application code
│   │   └── main.py
│   ├── model/                # Trained machine learning models
│   │   └── model_smote.pkl
│   ├── preprocessing/        # (Optional) Reusable preprocessing functions
│   └── utils/                # (Optional) General utility functions
├── frontend/                 # React frontend application code (planned)
├── docker/                   # Dockerization files
├── reports/                  # Generated reports (e.g., PDF)
├── .gitignore                # Git ignore file
├── README.md                 # This project overview
└── requirements.txt          # Python dependencies

Current Status & Progress
We have successfully:

Loaded and preprocessed the SECOM wafer dataset, handling missing values and non-numeric columns.

Addressed a critical issue with the target variable loading, ensuring correct numerical labels.

Trained an initial XGBoost classification model.

Applied SMOTE (Synthetic Minority Over-sampling Technique) to the training data to mitigate the severe class imbalance, showing some improvement in minority class recall.

Set up the basic FastAPI microservice (src/api/main.py) to serve model predictions and SHAP values.

Established a professional single-page landing website for the project.

Next Steps
Our immediate next steps include:

Testing the FastAPI Endpoint: Thoroughly test the /score endpoint to ensure it correctly receives data, makes predictions, and returns responses.

Developing the React Frontend: Build the user interface to interact with the API, display wafer heatmaps, and visualize insights.

Deployment: Containerize the application with Docker and deploy it to Azure App Service.

Model Refinement: Continuously improve model performance, especially for the minority class, through further hyperparameter tuning, exploring advanced techniques, or more data.

Getting Started (Local Development)
Clone the repository:

git clone <your-repo-url>
cd ai_yield_analytics

Set up Python environment:

# Using venv
python -m venv venv
.\venv\Scripts\activate # On Windows
source venv/bin/activate # On Linux/macOS

# Install dependencies
pip install -r requirements.txt

(Ensure requirements.txt is up-to-date with pandas, numpy, scikit-learn, xgboost, imbalanced-learn, fastapi, uvicorn, python-multipart, shap, matplotlib, seaborn).

Download SECOM data: Place secom.data and secom_labels.data into data/raw/.

Run Data Exploration: Execute notebooks/data_exploration.ipynb to preprocess data and save secom_preprocessed.csv.

Run Model Training: Execute notebooks/model_training.ipynb to train and save model_smote.pkl.

Run FastAPI server:

uvicorn src.api.main:app --reload

Access the Website: Open frontend/index.html in your browser. (Note: This is the static landing page, not the interactive React app yet).

Contact
If you have the SECOM data and would like to collaborate or check our solution, please feel free to contact us at: contact@aiyieldanalytics.com
