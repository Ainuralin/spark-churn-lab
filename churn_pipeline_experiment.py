from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# -----------------------------
# 1. Spark Session
# -----------------------------
spark = SparkSession.builder.appName("CustomerChurnPipelineExperiment").getOrCreate()

# -----------------------------
# 2. Load Data from HDFS
# -----------------------------
data = spark.read.csv(
    "hdfs:///user/hadoop/churn_input/Churn_Modelling.csv",
    header=True,
    inferSchema=True
)

# -----------------------------
# 3. Define Categorical Encoding
# -----------------------------
geo_indexer = StringIndexer(inputCol="Geography", outputCol="GeographyIndex")
gender_indexer = StringIndexer(inputCol="Gender", outputCol="GenderIndex")

encoder = OneHotEncoder(
    inputCols=["GeographyIndex", "GenderIndex"],
    outputCols=["GeographyVec", "GenderVec"]
)

# -----------------------------
# 4. Define Features
# -----------------------------
numerical_cols = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
categorical_cols = ["GeographyVec", "GenderVec"]

# Full features (for Logistic Regression and Random Forest)
all_features = numerical_cols + categorical_cols

# Features without categorical (for Feature Ablation)
numeric_only_features = numerical_cols

# -----------------------------
# 5. Assemble and Scale
# -----------------------------
assembler_all = VectorAssembler(inputCols=all_features, outputCol="features")
assembler_numeric = VectorAssembler(inputCols=numeric_only_features, outputCol="features")

scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")

# -----------------------------
# 6. Define Models
# -----------------------------
lr = LogisticRegression(labelCol="Exited", featuresCol="scaledFeatures")
rf = RandomForestClassifier(labelCol="Exited", featuresCol="scaledFeatures", numTrees=50, maxDepth=5)

# -----------------------------
# 7. Define Pipelines
# -----------------------------
# Logistic Regression full
pipeline_lr = Pipeline(stages=[geo_indexer, gender_indexer, encoder, assembler_all, scaler, lr])

# Random Forest full
pipeline_rf = Pipeline(stages=[geo_indexer, gender_indexer, encoder, assembler_all, scaler, rf])

# Logistic Regression numeric only (Feature Ablation)
pipeline_lr_numeric = Pipeline(stages=[assembler_numeric, scaler, lr])

# -----------------------------
# 8. Fit and Evaluate
# -----------------------------
evaluator = MulticlassClassificationEvaluator(
    labelCol="Exited", predictionCol="prediction", metricName="accuracy"
)

# --- Logistic Regression ---
model_lr = pipeline_lr.fit(data)
predictions_lr = model_lr.transform(data)
accuracy_lr = evaluator.evaluate(predictions_lr)
print("Logistic Regression (all features) Accuracy:", accuracy_lr)

# --- Random Forest ---
model_rf = pipeline_rf.fit(data)
predictions_rf = model_rf.transform(data)
accuracy_rf = evaluator.evaluate(predictions_rf)
print("Random Forest (all features) Accuracy:", accuracy_rf)

# --- Logistic Regression without categorical (Feature Ablation) ---
model_lr_numeric = pipeline_lr_numeric.fit(data)
predictions_lr_numeric = model_lr_numeric.transform(data)
accuracy_lr_numeric = evaluator.evaluate(predictions_lr_numeric)
print("Logistic Regression (numeric only) Accuracy:", accuracy_lr_numeric)

# -----------------------------
# 9. Show Sample Predictions
# -----------------------------
print("\nSample Predictions (Logistic Regression full):")
predictions_lr.select("Exited", "prediction", "probability").show(10)

# -----------------------------
# 10. Stop Spark Session
# -----------------------------
spark.stop()
