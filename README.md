# Spark ML Pipeline for Customer Churn Prediction on Amazon EMR

## Distributed Computing Lab 6

**Student:** Ali Ainur  
**Platform:** Amazon EMR (Spark)  
**Dataset:** Bank Customer Churn Dataset (Kaggle)  
**Repository:** [github.com/Ainuralin/spark-churn-lab](https://github.com/Ainuralin/spark-churn-lab)

---

## 📌 DATASET DESCRIPTION

**Source:**  
[https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)

**Description:**  
The dataset contains information about bank customers and their accounts. It includes both demographic and financial details to predict whether a customer will leave the bank (churn).

**Selected Features:**

| Feature | Type | Description |
|---------|------|-------------|
| CreditScore | Numerical | Customer's creditworthiness score |
| Geography | Categorical | Country (France, Spain, Germany) |
| Gender | Categorical | Male or Female |
| Age | Numerical | Customer's age in years |
| Balance | Numerical | Amount of money in customer's account |
| NumOfProducts | Numerical | Number of bank products used |
| **Exited (Target)** | **Binary** | **0 = Stayed, 1 = Churned** |

**Size:** 10,000 records

---

## 📌 STEP-BY-STEP WORKFLOW

### 1. EMR CLUSTER SETUP
- Created EMR cluster with 1 master node (m4.large) and 2 core nodes (m4.large)
- Installed Hadoop and Spark applications
- Configured security groups and IAM roles
- Connected to master node via SSH

### 2. DATA PREPARATION
- Downloaded Bank Customer Churn dataset from Kaggle
- Uploaded CSV file to EMR master node using SCP
- Copied dataset to HDFS distributed storage
- Verified successful upload with HDFS ls command

### 3. PIPELINE DEVELOPMENT
Created three Python scripts:

**a) churn_pipeline.py** - Initial Logistic Regression pipeline
- Load data from HDFS
- Encode categorical features (Geography, Gender)
- Assemble feature vectors
- Scale features using StandardScaler
- Train Logistic Regression model
- Evaluate with accuracy metric

**b) churn_pipeline_experiment.py** - Model comparison experiment
- Same pipeline structure with different classifiers
- Experiment 1: Logistic Regression (baseline)
- Experiment 2: Random Forest (non-linear ensemble)
- Experiment 3: Logistic Regression without categorical features (ablation study)

### 4. EXECUTION
```bash
# First run - test Logistic Regression pipeline
spark-submit --master yarn --deploy-mode client churn_pipeline.py

# Second run - run all three experiments
spark-submit --master yarn --deploy-mode client churn_pipeline_experiment.py
```

### 5. RESULTS COLLECTION
- Recorded accuracy scores for all three models
- Compared performance between linear and ensemble methods
- Analyzed impact of categorical feature removal

---

## 📌 EMR CLUSTER SETUP COMMANDS

**SSH to Master Node:**
```bash
ssh -i labsuser.pem hadoop@ec2-107-22-146-219.compute-1.amazonaws.com
```

---

## 📌 DATA UPLOAD COMMANDS

**Local Machine:**
```bash
scp -i labsuser.pem Churn_Modelling.csv hadoop@ec2-107-22-146-219.compute-1.amazonaws.com:/home/hadoop/
```

**EMR Master Node:**
```bash
hdfs dfs -mkdir -p /user/hadoop/churn_input
hdfs dfs -put Churn_Modelling.csv /user/hadoop/churn_input/
hdfs dfs -ls /user/hadoop/churn_input/
```

---

## 📌 SPARK-SUBMIT COMMANDS

**Logistic Regression (test):**
```bash
spark-submit --master yarn --deploy-mode client churn_pipeline.py
```

**All Experiments (LR, RF, LR no categories):**
```bash
spark-submit --master yarn --deploy-mode client churn_pipeline_experiment.py
```

---

## 📌 RESULTS

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 0.7929 |
| Random Forest | 0.8413 |
| Logistic w/o categories | 0.7797 |

---

## 📌 CONCLUSION

- **Random Forest (84.13%)** performed better than Logistic Regression (79.29%) because it captures non-linear patterns in customer behavior
- **Categorical features** improved accuracy by ~1.32%, showing Geography and Gender are relevant predictors
- **Spark on EMR** successfully distributed the workload across 3 nodes, making the pipeline scalable for larger datasets
- The **ML Pipeline** approach ensured consistent data transformations between training and testing

---

## 📌 REFERENCES

- Article: [Implementing Machine Learning Pipelines with Apache Spark](https://www.kdnuggets.com/implementing-machine-learning-pipelines-with-apache-spark)
- Dataset: [Bank Customer Churn Dataset - Kaggle](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)

