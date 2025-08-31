# AI-Assistant-Usage

##  Project Overview
This project focuses on building and evaluating machine learning classification models to predict the target variable from a given dataset.  
The main goal was to compare multiple models and identify the one that performs best based on accuracy and other classification metrics.

---

##  Steps Taken
1. **Data Preprocessing**
   - Handled missing values  
   - Encoded categorical variables  
   - Scaled numerical features using `StandardScaler`  

2. **Exploratory Data Analysis (EDA)**
   - Visualized class distribution  
   - Analyzed feature correlations  
   - Checked for outliers and patterns  

3. **Model Training**
   - Tried multiple classification algorithms:  
     - Logistic Regression  
     - Decision Tree Classifier  
     - Random Forest Classifier  
     - Support Vector Machine (SVM)  
   - Used **train-test split** for evaluation  

4. **Model Evaluation**
   - Compared models using:  
     - Accuracy  
     - Precision  
     - Recall  
     - F1-score
       
 ## Models Implemented
We trained and compared the following classification models:
- Logistic Regression  
- Decision Tree Classifier (with GridSearchCV hyperparameter tuning)  
- Random Forest Classifier (tuned `n_estimators`, `max_depth`)  
- Naive Bayes  
- K-Nearest Neighbors (KNN)  
- Gradient Boosting  
- XGBoost
  
##  Model Tuning
- **Decision Tree:** Tuned using `GridSearchCV` (parameters: `max_depth`, `min_samples_split`).  
- **Random Forest:** Tuned using `GridSearchCV` (parameters: `n_estimators`, `max_depth`).  
- **Cross-validation:** Applied (5-fold) to ensure robust evaluation.  
---

##  Best Model
After experimentation, the **Logistics Regression** performed the best with the following classification report

Accuracy: 0.7425

Confusion Matrix:
 [[ 274  313]
 [ 202 1211]]

Classification Report:
               precision    recall  f1-score   support

       False       0.58      0.47      0.52       587
        True       0.79      0.86      0.82      1413

    accuracy                           0.74      2000
   macro avg       0.69      0.66      0.67      2000
weighted avg       0.73      0.74      0.73      2000



