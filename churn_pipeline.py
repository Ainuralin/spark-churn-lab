from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# 1. Create Spark session
spark = SparkSession.builder.appName("CustomerChurnPipeline").getOrCreate()

# 2. Load data from HDFS
data = spark.read.csv(
    "hdfs:///user/hadoop/churn_input/Churn_Modelling.csv",
    header=True,
    inferSchema=True
)

# 3. Select needed columns
data = data.select(
    "CreditScore", "Geography", "Gender", "Age",
    "Tenure", "Balance", "NumOfProducts",
    "EstimatedSalary", "Exited"
)

# 4. Categorical encoding
geo_indexer = StringIndexer(inputCol="Geography", outputCol="GeographyIndex")
gender_indexer = StringIndexer(inputCol="Gender", outputCol="GenderIndex")

encoder = OneHotEncoder(
    inputCols=["GeographyIndex", "GenderIndex"],
    outputCols=["GeographyVec", "GenderVec"]
)

# 5. Assemble features
assembler = VectorAssembler(
    inputCols=[
        "CreditScore", "Age", "Tenure", "Balance",
        "NumOfProducts", "EstimatedSalary",
        "GeographyVec", "GenderVec"
    ],
    outputCol="features"
)

# 6. Scale features
scaler = StandardScaler(
    inputCol="features",
    outputCol="scaledFeatures"
)

# 7. Model
lr = LogisticRegression(
    labelCol="Exited",
    featuresCol="scaledFeatures"
)

# 8. Build pipeline
pipeline = Pipeline(stages=[
    geo_indexer,
    gender_indexer,
    encoder,
    assembler,
    scaler,
    lr
])

# 9. Train model
model = pipeline.fit(data)

# 10. Make predictions
predictions = model.transform(data)

predictions.select("Exited", "prediction", "probability").show(10)

# 11. Evaluate
evaluator = MulticlassClassificationEvaluator(
    labelCol="Exited",
    predictionCol="prediction",
    metricName="accuracy"
)

accuracy = evaluator.evaluate(predictions)
print("Accuracy:", accuracy)

spark.stop()
