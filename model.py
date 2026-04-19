# Step 1: Imports
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

import warnings
warnings.filterwarnings("ignore")

# Step 2: Load & Clean Data
df = pd.read_csv('loan_dataset.csv')

# Clean column names
df.columns = df.columns.str.strip()

# Drop ID column
df = df.drop("Loan_ID", axis=1)


# Step 3: Define Column Groups
flt_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
int_cols = ['Credit_History', 'Loan_Amount_Term']
cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']


# Step 4: Fix Data Types
def fix_dtypes(df):
    for col in df.columns:
        
        if col in flt_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        elif col in int_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        
        elif col in cat_cols:
            df[col] = df[col].astype("category")
    
    return df

df = fix_dtypes(df)


# Step 5: Target Encoding (CRITICAL)
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
df = df.dropna(subset=['Loan_Status'])


# Step 6: Split Data (Stratified)
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Step 7: Create Pipelines
# Float Pipeline
flt_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Int Pipeline
int_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent"))
])

# Categorical Pipeline
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Step 8: ColumnTransformer
preprocessor = ColumnTransformer([
    ("flt", flt_pipeline, flt_cols),
    ("int", int_pipeline, int_cols),
    ("cat", cat_pipeline, cat_cols)
])


# Step 9: Final Pipeline
pipe = Pipeline([
    ("preprocessing", preprocessor),
    ("model", LogisticRegression(max_iter=5000, class_weight='balanced', random_state=42))
])


# Step 10: Hyperparameter Grid
params = {
    "model__penalty": ['l1', 'l2'],
    "model__C": [100, 10, 1, 0.1, 0.01],
    "model__solver": ['liblinear', 'saga']
}

scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall'
}


# Step 11: Cross Validation Setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# Step 12: GridSearchCV
grid = GridSearchCV(
    estimator=pipe,
    param_grid=params,
    scoring=scoring,
    refit='accuracy',
    cv=cv,
    n_jobs=-1,
    verbose=1
)


# Step 13: Train Model
grid.fit(X_train, y_train)


# Step 14: Predictions
y_pred = grid.predict(X_test)


# Step 15: Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

y_prob = grid.predict_proba(X_test)[:,1]
print("ROC-AUC:", roc_auc_score(y_test, y_prob))


import pickle

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(grid.best_estimator_, f)