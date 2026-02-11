# Bank Customer Churn Prediction - Spark ML Pipeline on EMR
# Student: Ali Ainur
# Distributed Computing Lab

===============================================================================
# DATASET DESCRIPTION
===============================================================================
# Name: Bank Customer Churn Dataset
# Source: Kaggle (https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)
# File: Churn_Modelling.csv
# Records: 10,000 bank customers
# Features:
#   - CreditScore    : Customer credit score (numeric)
#   - Geography      : Country (France/Spain/Germany)
#   - Gender         : Male/Female
#   - Age           : Customer age (numeric)
#   - Tenure        : Years with bank (numeric)
#   - Balance       : Account balance (numeric)
#   - NumOfProducts  : Banking products used (numeric)
#   - EstimatedSalary: Customer salary (numeric)
#   - Exited        : Target variable (0=Stayed, 1=Churned)

===============================================================================
# PREREQUISITES
===============================================================================
# 1. AWS EMR Cluster with:
#    - Hadoop
#    - Spark
#    - 1 master node (m4.large)
#    - 2 core nodes (m4.large)
#
# 2. Dataset uploaded to EMR master node:
#    scp -i key.pem Churn_Modelling.csv hadoop@<master-dns>:/home/hadoop/
#
# 3. Dataset copied to HDFS:
#    hdfs dfs -mkdir -p /user/hadoop/churn_input
#    hdfs dfs -put Churn_Modelling.csv /user/hadoop/churn_input/

===============================================================================
# PIPELINE COMPONENTS
===============================================================================
# 1. Data Loading      : Load CSV from HDFS
# 2. Encoding         : StringIndexer + OneHotEncoder for Geography/Gender
# 3. Feature Assembly : VectorAssembler (all features)
# 4. Scaling         : StandardScaler
# 5. Model Training   : Logistic Regression / Random Forest
# 6. Prediction      : Transform test data
# 7. Evaluation      : Accuracy metric

===============================================================================
# SPARK SUBMIT COMMANDS
===============================================================================

# 1. LOGISTIC REGRESSION PIPELINE
# ---------------------------------
spark-submit \
  --master yarn \
  --deploy-mode client \
  --name "Churn_Prediction_LR" \
  churn_pipeline.py

# Expected output: Accuracy: ~0.7929

# 2. RANDOM FOREST PIPELINE  
# ---------------------------------
# Edit churn_pipeline.py: Replace LogisticRegression with RandomForestClassifier
# from pyspark.ml.classification import RandomForestClassifier
# rf = RandomForestClassifier(labelCol="Exited", featuresCol="scaledFeatures")

spark-submit \
  --master yarn \
  --deploy-mode client \
  --name "Churn_Prediction_RF" \
  churn_pipeline_rf.py

# Expected output: Accuracy: ~0.8413

# 3. LOGISTIC REGRESSION (NO CATEGORICAL FEATURES)
# ---------------------------------
# Edit churn_pipeline.py: Remove GeographyVec, GenderVec from VectorAssembler

spark-submit \
  --master yarn \
  --deploy-mode client \
  --name "Churn_Prediction_LR_NoCat" \
  churn_pipeline_no_cat.py

# Expected output: Accuracy: ~0.7797

===============================================================================
# RESULTS SUMMARY
===============================================================================
# Model                          Accuracy
# ----------------------------------------
# Logistic Regression            0.7929
# Random Forest                 0.8413  
# Logistic Regression (no cats)  0.7797
#
# Conclusion: Random Forest performs best for this dataset.
# Categorical features improve model accuracy by ~1.3%.

===============================================================================
# FILE STRUCTURE
===============================================================================
# spark-churn-lab/
# │
# ├── churn_pipeline.py           # Logistic Regression pipeline
# ├── churn_pipeline_rf.py        # Random Forest pipeline
# ├── churn_pipeline_no_cat.py    # LR without categorical features
# ├── README.md                   # This file
# └── Churn_Modelling.csv         # Dataset (local copy)

===============================================================================
# VERIFY HDFS DATA
===============================================================================
hdfs dfs -ls /user/hadoop/churn_input/
hdfs dfs -cat /user/hadoop/churn_input/Churn_Modelling.csv | head -5

===============================================================================
# MONITOR SPARK JOB
===============================================================================
# YARN ResourceManager: http://<master-dns>:8088
# Spark UI: http://<master-dns>:4040

===============================================================================
# GITHUB REPOSITORY
===============================================================================
# https://github.com/Ainuralin/spark-churn-lab
#
# Clone repository:
# git clone https://github.com/Ainuralin/spark-churn-lab.git
# cd spark-churn-lab

===============================================================================
# AUTHOR
===============================================================================
# Name: Ali Ainur
# Course: Distributed Computing
# Lab: Spark ML Pipeline on Amazon EMR
# Date: February 2026
