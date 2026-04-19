
# 🏦 Loan Approval Prediction System (ML + FastAPI + Streamlit)

An end-to-end **Machine Learning system** that predicts whether a loan application will be approved or not based on applicant details.

Built using **Scikit-learn Pipelines**, **Logistic Regression**, and deployed with **FastAPI** and **Streamlit** for real-time predictions.

---

## 🚀 Features

### 🔍 Loan Approval Prediction

* Enter applicant details
* Get instant prediction:
  * Loan Approved
  * Loan Not Approved
* Probability score for confidence

### ⚙️ End-to-End ML Pipeline

* Data preprocessing using pipelines
* Automatic handling of:
  * Missing values
  * Feature scaling
  * Encoding categorical variables

### 📊 Hyperparameter Optimization

* GridSearchCV for best model selection
* Multiple combinations of:
  * Regularization
  * Solvers

### 🧠 Robust Model Training

* Logistic Regression with class balancing
* Stratified cross-validation
* Prevents bias in predictions

### ⚡ API Deployment

* FastAPI backend
* Real-time prediction endpoint
* JSON-based request/response

### 🎨 Interactive UI

* Streamlit frontend
* Easy-to-use input form
* Instant results display

---

## 🏗️ Architecture

```
User Input (Streamlit UI)
        ↓
FastAPI Backend (/predict)
        ↓
Preprocessing Pipeline
        ↓
Logistic Regression Model
        ↓
Prediction + Probability
        ↓
Displayed in UI
```

---

## 🧠 Tech Stack

* **Frontend**: Streamlit  
* **Backend API**: FastAPI  
* **ML Framework**: Scikit-learn  
* **Data Processing**: Pandas, NumPy  
* **Model Persistence**: Pickle  

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/loan-prediction.git
cd loan-prediction
```

---

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

### Step 1: Train the Model

```bash
python model_training.py
```

---

### Step 2: Start FastAPI Server

```bash
uvicorn app:app --reload
```

---

### Step 3: Launch Streamlit UI

```bash
streamlit run streamlit_app.py
```

---

## 💡 Usage

### Step 1: Enter Applicant Details

* Income
* Loan amount
* Credit history
* Personal details

---

### Step 2: Get Prediction

Output includes:

* Loan Approval Status  
* Probability Score  

---

## 🧪 Example Inputs

```
ApplicantIncome: 5000
CoapplicantIncome: 2000
LoanAmount: 150
Credit_History: 1
Loan_Amount_Term: 360
Gender: Male
Married: Yes
Dependents: 0
Education: Graduate
Self_Employed: No
Property_Area: Urban
```

---

## ⚠️ Limitations

* Model performance depends on dataset quality  
* Limited to structured tabular data  
* Does not include external financial factors  

---

## 📈 Future Improvements

* [ ] Deploy on cloud (AWS / GCP / Render)  
* [ ] Add Docker support  
* [ ] Improve UI/UX  
* [ ] Add model explainability (SHAP)  
* [ ] Real-time data integration  

---

## 🧠 What This Project Demonstrates

* End-to-end ML pipeline design  
* Feature engineering with pipelines  
* Hyperparameter tuning  
* API deployment using FastAPI  
* Frontend integration with Streamlit  
* Production-ready ML workflow  

---

## 🤝 Contributing

Contributions are welcome!  
Feel free to open issues or submit pull requests.

---


## 🙌 Acknowledgements

* Scikit-learn  
* FastAPI  
* Streamlit  
* Pandas  
* NumPy  

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!

---
