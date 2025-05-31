import pyspark
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("cw3_Q4").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

df = spark.read.csv("/users/sx1716/cw3/ObesityDataSet_raw_and_data_sinthetic.csv", header=True, inferSchema=True)
df = df.dropna()

# -------------------------------
# EDA
# -------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import chi2_contingency, f_oneway

# Convert Spark to pandas for EDA
pdf = df.toPandas()

# Clean column names 
pdf.columns = pdf.columns.str.strip()

# --------------------------------
# Cramér’s V for Categorical vs NObeyesdad
# --------------------------------
def cramers_v(confusion_matrix):
    """Compute Cramér's V statistic for categorical-categorical association."""
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - (k-1)*(r-1)/(n-1))
    rcorr = r - (r-1)**2/(n-1)
    kcorr = k - (k-1)**2/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

categorical_cols = ["Gender", "family_history_with_overweight", "FAVC", "CAEC", 
                    "SMOKE", "SCC", "CALC", "MTRANS", "NObeyesdad"]

# Compute Cramér’s V of each categorical feature
target_col = "NObeyesdad"
for cat_col in categorical_cols:
    if cat_col == target_col:
        continue
    # Create a contingency table
    contingency_table = pd.crosstab(pdf[cat_col], pdf[target_col])
    cv = cramers_v(contingency_table.values)
    print(f"Cramér’s V between {cat_col} and {target_col}: {cv:.4f}")

# --------------------------------
# ANOVA for Numeric vs NObeyesdad
# --------------------------------
numeric_cols = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]

for num_col in numeric_cols:
    # Collect the numeric values for each obesity category
    groups = [group[num_col].values for _, group in pdf.groupby(target_col)]
    # Perform one-way ANOVA
    f_stat, p_val = f_oneway(*groups)
    print(f"ANOVA for {num_col} across {target_col}: F={f_stat:.4f}, p={p_val:.4e}")

# Class distribution count
class_counts = pdf['NObeyesdad'].value_counts()
class_proportions = pdf['NObeyesdad'].value_counts(normalize=True) * 100

print("Class Counts:")
print(class_counts)
print("\nClass Proportions (%):")
print(class_proportions)

# ---------------
# EDA plots
# --------------
plt.figure(figsize=(10, 5))
sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis")
plt.title("Obesity Class Distribution")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Distribution of weight
plt.figure(figsize=(10, 5))
sns.histplot(data=pdf, x="Weight", bins=30)
plt.title("Distribution of Weight")
plt.xlabel("Weight")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

# Boxplot of weight by Obesity Category
plt.figure(figsize=(10, 5))
sns.boxplot(data=pdf, x="NObeyesdad", y="Weight", order=pdf['NObeyesdad'].value_counts().index)
plt.title("Weight Distribution by Obesity Category")
plt.xlabel("Obesity Category")
plt.ylabel("Weight")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Correlation Heatmap for numeric features
numeric_cols = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
corr_matrix = pdf[numeric_cols].corr()
plt.figure(figsize=(8, 5))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap (only numeric features)")
plt.tight_layout()
plt.show()

# -------------------------------
# Feature and LR, RF, DT classification
# -------------------------------
from pyspark.ml.feature import StringIndexer, VectorAssembler, PCA
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.functions import when, udf, col
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.ml.feature import IndexToString

# Define categorical columns and shared transformations
categorical_cols = ["Gender", "family_history_with_overweight", "FAVC", "CAEC", "SMOKE", "SCC", "CALC", "MTRANS"]
indexers = [StringIndexer(inputCol=col, outputCol=col+"_indexed", handleInvalid="keep") for col in categorical_cols]

# BMI-based class label mapping
bmi_order = ["Insufficient_Weight", "Normal_Weight", "Overweight_Level_I", 
             "Overweight_Level_II", "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"]
bmi_label_to_index = {label: i for i, label in enumerate(bmi_order)}
index_to_label = {i: label for label, i in bmi_label_to_index.items()}

df = df.withColumn("label", 
    when(col("NObeyesdad") == "Insufficient_Weight", 0)
    .when(col("NObeyesdad") == "Normal_Weight", 1)
    .when(col("NObeyesdad") == "Overweight_Level_I", 2)
    .when(col("NObeyesdad") == "Overweight_Level_II", 3)
    .when(col("NObeyesdad") == "Obesity_Type_I", 4)
    .when(col("NObeyesdad") == "Obesity_Type_II", 5)
    .when(col("NObeyesdad") == "Obesity_Type_III", 6)
)

# Assemble features
input_features = numeric_cols + [col+"_indexed" for col in categorical_cols]
assembler = VectorAssembler(inputCols=input_features, outputCol="features")
shared_stages = indexers + [assembler]

# Clean indexed column conflicts
for c in df.columns:
    if c.endswith("_indexed"):
        df = df.drop(c)
# Split
train_data, test_data = df.randomSplit([0.8, 0.2], seed=1234)
train_data.cache()
test_data.cache()

# -------------------------------
# Model definitions and pipelines
# -------------------------------

# Logistic Regression
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10, family="multinomial")
lr_pipeline = Pipeline(stages=shared_stages + [lr])
lr_paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.05, 0.01, 0.1]).addGrid(lr.elasticNetParam, [0.1, 0.5, 1]).addGrid(lr.maxIter, [10, 20, 50]).build()
lr_cv = CrossValidator(estimator=lr_pipeline, estimatorParamMaps=lr_paramGrid,
                       evaluator=MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy"), numFolds=5)
lr_model = lr_cv.fit(train_data)
class_labels = bmi_order
lr_stage = lr_model.bestModel.stages[-1]         
lr_predictions = lr_model.transform(test_data)

# Print optimal logistic regression parameters
print("Best LR Params:")
print(f"  regParam: {lr_stage._java_obj.getRegParam()}")
print(f"  elasticNetParam: {lr_stage._java_obj.getElasticNetParam()}")
print(f"  maxIter: {lr_stage._java_obj.getMaxIter()}")

# Random Forest 
rf = RandomForestClassifier(featuresCol="features", labelCol="label")
rf_pipeline = Pipeline(stages=shared_stages + [rf])
rf_paramGrid = ParamGridBuilder().addGrid(rf.numTrees, [50, 100, 150]).addGrid(rf.maxDepth, [5, 10, 15]).build()
rf_cv = CrossValidator(estimator=rf_pipeline,
                       estimatorParamMaps=rf_paramGrid,
                       evaluator=MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy"),
                       numFolds=5)
rf_model = rf_cv.fit(train_data)
rf_predictions = rf_model.transform(test_data)

# Print best RF parameters
print("Best Random Forest Params:", rf_model.bestModel.stages[-1].extractParamMap())

# Decision tree classifier
dt = DecisionTreeClassifier(featuresCol="features", labelCol="label")
dt_pipeline = Pipeline(stages=shared_stages + [dt])
dt_paramGrid = ParamGridBuilder().addGrid(dt.maxDepth, [3, 5, 7]).addGrid(dt.impurity, ["gini", "entropy"]).build()
dt_cv = CrossValidator(estimator=dt_pipeline,estimatorParamMaps=dt_paramGrid,evaluator=MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy"), numFolds=5)
dt_model = dt_cv.fit(train_data)

dt_predictions = dt_model.transform(test_data)

# Accuracy， F1 score
def evaluate_model(predictions, model_name):
    evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", metricName="f1")

    accuracy = evaluator_acc.evaluate(predictions)
    f1_score = evaluator_f1.evaluate(predictions)
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print(f"{model_name} F1 Score: {f1_score:.4f}")

# Evaluate all models
evaluate_model(lr_predictions, "Logistic Regression")
evaluate_model(rf_predictions, "Random Forest")
evaluate_model(dt_predictions, "Decision Tree")

from sklearn.metrics import confusion_matrix

# Confusion matrix for RF
rf_preds_pd = rf_predictions.select("label", "prediction").toPandas()
cm = confusion_matrix(rf_preds_pd["label"], rf_preds_pd["prediction"])
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest Confusion Matrix")
plt.show()

# Get feature importances from best RF model
rf_stage = rf_model.bestModel.stages[-1]
importances = rf_stage.featureImportances.toArray()
features = input_features

# Convert to DataFrame for plotg
fi_df = pd.DataFrame({'Feature': features, 'Importance': importances})
fi_df = fi_df.sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=fi_df.head(15), x='Importance', y='Feature')
plt.title("Top Feature Importances (Random Forest)")
plt.tight_layout()
plt.show()

# Confusion matrix for LR
lr_preds_pd = lr_predictions.select("label", "prediction").toPandas()
cm_lr = confusion_matrix(lr_preds_pd["label"], lr_preds_pd["prediction"])
plt.figure(figsize=(8,6))
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Greens")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Logistic Regression Confusion Matrix")
plt.show()

# Confusion matrix for DT
dt_preds_pd = dt_predictions.select("label", "prediction").toPandas()
cm_dt = confusion_matrix(dt_preds_pd["label"], dt_preds_pd["prediction"])
plt.figure(figsize=(8,6))
sns.heatmap(cm_dt, annot=True, fmt="d", cmap="Oranges")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Decision Tree Confusion Matrix")
plt.show()