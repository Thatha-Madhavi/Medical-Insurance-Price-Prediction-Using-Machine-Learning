# Medical-Insurance-Price-Prediction-Using-Machine-Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" />
  <img src="https://img.shields.io/badge/Machine Learning-Regression-orange" />
  <img src="https://img.shields.io/badge/Notebook-Jupyter-yellow" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
  <img src="https://img.shields.io/badge/Status-Active-brightgreen" />
</p>

This project is my complete end-to-end implementation of a machine learning regression model that predicts medical insurance charges based on user details like age, BMI, smoking habits, number of children, and region.
I built this project to understand how insurance pricing works and to practice a full ML workflow — from raw data to a ready-to-use prediction model.

## 📌 1. Introduction

Healthcare insurance companies determine premium prices based on personal and lifestyle factors.
However, these prices are often complex and difficult to estimate manually.

### Goal of this project:

Build a machine learning model that can predict insurance cost using demographic and health-related variables.

This is a supervised regression problem.

## 📊 2. Dataset Description

The dataset contains information about individuals and the insurance cost they paid.
It includes the following features:

| Column       | Description                                            |
| ------------ | ------------------------------------------------------ |
| **age**      | Age of the person                                      |
| **sex**      | Male / Female                                          |
| **bmi**      | Body Mass Index — indicates health & obesity           |
| **children** | Number of dependent children                           |
| **smoker**   | Whether the person is a smoker (Yes/No)                |
| **region**   | US region (southeast, southwest, northeast, northwest) |
| **charges**  | Final insurance cost (Target Variable)                 |

The feature charges is what we want to predict.

🔍 3. Project Pipeline (Step-by-Step Explanation)

This is the heart of your project:

## 🔄 Project Workflow

           ┌──────────────────┐
           │  Load Dataset     │
           └─────────┬────────┘
                     │
                     ▼
           ┌──────────────────┐
           │  Exploratory      │
           │  Data Analysis    │
           └─────────┬────────┘
                     │
                     ▼
           ┌──────────────────┐
           │ Data Preprocessing│
           │ (Encoding, Split) │
           └─────────┬────────┘
                     │
                     ▼
           ┌──────────────────┐
           │  Model Training   │
           │ (LR, RF, etc.)    │
           └─────────┬────────┘
                     │
                     ▼
           ┌──────────────────┐
           │ Model Evaluation  │
           │ (MAE, RMSE, R²)   │
           └─────────┬────────┘
                     │
                     ▼
           ┌──────────────────┐
           │ Save & Deploy     │
           │ Model (pickle)    │
           └──────────────────┘

### ✔️ Step 1: Import Libraries

#### Load essential libraries for:

- Data processing → pandas, numpy

- Visualization → matplotlib, seaborn

- Machine learning → scikit-learn

<img width="1216" height="321" alt="Screenshot 2025-11-30 164957" src="https://github.com/user-attachments/assets/9c5bb18d-9569-4030-a4d0-9030a8d5fceb" />

### ✔️ Step 2: Load and Inspect Data
<img width="514" height="110" alt="Screenshot 2025-11-30 165209" src="https://github.com/user-attachments/assets/4c5dbcdb-95bd-4e07-b15b-94cc79c2ddd8" />

**Output**

<img width="878" height="634" alt="Screenshot 2025-11-30 165247" src="https://github.com/user-attachments/assets/7af394ac-f17f-422c-9383-2948d6b6e30a" />

We check:

- Data types

- Missing values

- Statistical summary

- Overall structure

💡 This step ensures the dataset is clean and ready for modeling.

### ✔️ Step 3: Exploratory Data Analysis (EDA)

EDA helps us understand patterns, relationships, and hidden insights.

#### 3.1 Univariate Analysis

- Age distribution

- Charges distribution

- BMI analysis

- Smokers vs Non-smokers count

#### 3.2 Bivariate Analysis

- Charges vs Age

- Charges vs BMI

- Charges vs Smoker

- Charges vs Region

##### 👉 Key Observations you can mention:

- Smokers have drastically higher insurance charges

- Higher BMI increases cost

- Age positively correlates with cost

 #### 3.3 Correlation Matrix

Visualization using heatmap helps identify highly important features.

### ✔️ Step 4: Data Preprocessing

#### 4.1 Encoding Categorical Variables

Machine Learning models require numeric values.

We convert:

- sex → male/female

- smoker → yes/no

- region → one-hot encoded


<img width="560" height="125" alt="Screenshot 2025-11-30 165806" src="https://github.com/user-attachments/assets/b7bac3e4-db3d-4d50-89f4-0dc0408e4eba" />

**Output**
<img width="562" height="394" alt="image" src="https://github.com/user-attachments/assets/17d9b6c1-373a-4535-b2b3-8832a84daff9" />

#### 4.2 Train-Test Split

<img width="1173" height="715" alt="Screenshot 2025-11-30 170449" src="https://github.com/user-attachments/assets/0a10b402-5054-48b0-89c5-e4cbb21f441b" />

### 🤖 5. Model Development (Main ML Logic)

I build multiple regression models and compare performance.

Models Used

- Linear Regression
- Support Vector Machine
- Random Forest Regressor
- Gradient Boost
- XGBoost
  
### ✔️ Step 6: Training Models
#### Linear Regression

<img width="1076" height="379" alt="image" src="https://github.com/user-attachments/assets/60a087bc-9215-4ae0-b55d-51fd4cd1986d" />

#### Support Vector Machine(SVM)

<img width="709" height="367" alt="Screenshot 2025-11-30 171159" src="https://github.com/user-attachments/assets/f5661b42-f208-45f1-9b79-909c9e343f59" />


#### Random Forest

<img width="887" height="887" alt="Screenshot 2025-11-30 171237" src="https://github.com/user-attachments/assets/de70f2a4-2abf-4a11-845f-332a4da9133c" />

#### Gradient Boost

<img width="1099" height="904" alt="Screenshot 2025-11-30 171517" src="https://github.com/user-attachments/assets/dff65a38-b0a7-469f-8e68-81e76aa33e6f" />

#### XGBoost

<img width="1138" height="872" alt="Screenshot 2025-11-30 171600" src="https://github.com/user-attachments/assets/a95c0bff-ed91-45ae-a8dd-6655646f53af" />

### 📊 Model Evaluation

| **Model**                  | **Train Accuracy (R²)** | **Test Accuracy (R²)** | **CV Score** |
| -------------------------- | ----------------------- | ---------------------- | ------------ |
| **Linear Regression**      | 0.729                   | 0.806                  | 0.747        |
| **Support Vector Machine** | -0.105                  | -0.134                 | 0.103        |
| **Random Forest**          | **0.974**               | **0.882**              | **0.836**    |
| **Gradient Boosting**      | 0.868                   | 0.901                  | 0.860        |
| **XGBoost**                | 0.870                   | **0.904**              | 0.860        |

#### Final Model 

As you see XGBoost is best. So, i identified the important features and here is the final model. 

<img width="989" height="528" alt="Screenshot 2025-11-30 172903" src="https://github.com/user-attachments/assets/02696c44-64f8-4861-b350-835d51922ccb" />

### Saving Model for Deployment
<img width="640" height="96" alt="Screenshot 2025-11-30 173028" src="https://github.com/user-attachments/assets/80fd16dc-849c-4d2a-bd16-81d3ca10acc4" />

### Predict on New Data

<img width="1498" height="255" alt="image" src="https://github.com/user-attachments/assets/60350cbe-25bb-42af-a9a5-aa35eeddca2e" />
🌐 10. Future Enhancements

- Add Flask / FastAPI Web App

- Deploy model to cloud (AWS / Render / Railway)

- Hyperparameter tuning (GridSearchCV)

- Add Streamlit UI for clean interface

- Improve model accuracy using XGBoost / Gradient Boosting

- Add SHAP explainability graphs


---
## ⭐ Acknowledgements
This project was built to practice real-world machine learning workflows and end-to-end model development.  
Inspired by healthcare cost analysis used by insurance companies.

---

## 📝 Contact
**Author:** Thatha Madhavi    
**GitHub:** [Thatha-Madhavi](https://github.com/Thatha-Madhavi)

Feel free to connect for collaboration or suggestions!

---
