import os
import tempfile
import shutil
temp_dir = os.path.join(os.getcwd(), 'spark_temp')
os.makedirs(temp_dir, exist_ok=True)
os.environ['SPARK_LOCAL_DIRS'] = temp_dir
os.environ['TMPDIR'] = temp_dir

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, mean
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, DecisionTreeClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

print("\nflight delay prediction - machine learning")
print("-" * 60)

# create spark session
spark = SparkSession.builder \
    .appName("Flight Delay ML") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

# reduce logging noise
spark.sparkContext.setLogLevel("ERROR")

print("spark session created")

# load data
data_path = "Airlines.csv"
print(f"loading: {data_path}")
data = spark.read.csv(data_path, header=True, inferSchema=True)

print(f"total records: {data.count():,}")
print(f"features: {len(data.columns)}")

# show sample
print("\nsample data:")
data.show(5)

# preprocessing
print("\n" + "-" * 60)
print("preprocessing data")
print("-" * 60)

# check missing values
from pyspark.sql.functions import isnan, count as spark_count
print("\nmissing values:")
data.select([spark_count(when(col(c).isNull(), c)).alias(c) for c in data.columns]).show()

# fill missing values
print("\nfilling missing values...")
numerical_columns = ['Time', 'Length']
for col_name in numerical_columns:
    mean_value = data.select(mean(col(col_name))).collect()[0][0]
    if mean_value is not None:
        data = data.fillna({col_name: mean_value})

categorical_columns = ['Airline', 'AirportFrom', 'AirportTo']
for col_name in categorical_columns:
    data = data.fillna({col_name: 'Unknown'})

print("missing values handled")

# create features
print("\ncreating features...")
data = data.withColumn('Hour', (col('Time') / 100).cast('int'))
data = data.withColumn('IsWeekend', when(col('DayOfWeek').isin([6, 7]), 1).otherwise(0))
data = data.withColumn('IsPeakHour', 
                       when(col('Hour').between(7, 9) | col('Hour').between(17, 19), 1).otherwise(0))
data = data.withColumn('FlightDurationCategory',
                       when(col('Length') < 100, 0)
                       .when(col('Length').between(100, 200), 1)
                       .when(col('Length').between(200, 300), 2)
                       .otherwise(3))

print("features created")

# encode categorical variables
print("\nencoding categorical variables...")
indexers = [
    StringIndexer(inputCol="Airline", outputCol="AirlineIndex", handleInvalid="keep"),
    StringIndexer(inputCol="AirportFrom", outputCol="AirportFromIndex", handleInvalid="keep"),
    StringIndexer(inputCol="AirportTo", outputCol="AirportToIndex", handleInvalid="keep")
]

for indexer in indexers:
    data = indexer.fit(data).transform(data)

# drop original columns
data = data.drop('Airline', 'AirportFrom', 'AirportTo', 'id', 'Flight', 'Time')

# create feature vector
feature_columns = [col for col in data.columns if col != 'Delay']
print(f"\nusing {len(feature_columns)} features")

assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
data = assembler.transform(data)

# select features and label
final_data = data.select("features", "Delay")
print("\nfeature vector created")

# train-test split
print("\n" + "-" * 60)
print("splitting data")
print("-" * 60)

train_data, test_data = final_data.randomSplit([0.8, 0.2], seed=42)
train_count = train_data.count()
test_count = test_data.count()

print(f"training set: {train_count:,} ({train_count/final_data.count()*100:.1f}%)")
print(f"testing set: {test_count:,} ({test_count/final_data.count()*100:.1f}%)")

# check class distribution
print("\nclass distribution in training set:")
train_data.groupBy("Delay").count().show()

# train models
print("\n" + "-" * 60)
print("training machine learning models")
print("-" * 60)

results = []

# setup evaluators
evaluator_acc = MulticlassClassificationEvaluator(labelCol="Delay", metricName="accuracy")
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="Delay", metricName="f1")
evaluator_precision = MulticlassClassificationEvaluator(labelCol="Delay", metricName="weightedPrecision")
evaluator_recall = MulticlassClassificationEvaluator(labelCol="Delay", metricName="weightedRecall")
evaluator_auc = BinaryClassificationEvaluator(labelCol="Delay", metricName="areaUnderROC")

# model 1: logistic regression
print("\n1. training logistic regression...")
lr = LogisticRegression(labelCol="Delay", featuresCol="features", maxIter=100, regParam=0.01)
lr_model = lr.fit(train_data)
lr_predictions = lr_model.transform(test_data)

lr_accuracy = evaluator_acc.evaluate(lr_predictions)
lr_f1 = evaluator_f1.evaluate(lr_predictions)
lr_precision = evaluator_precision.evaluate(lr_predictions)
lr_recall = evaluator_recall.evaluate(lr_predictions)
lr_auc = evaluator_auc.evaluate(lr_predictions)

print(f"   accuracy: {lr_accuracy*100:.2f}%")
print(f"   f1-score: {lr_f1:.4f}")
print(f"   precision: {lr_precision:.4f}")
print(f"   recall: {lr_recall:.4f}")
print(f"   auc: {lr_auc:.4f}")

results.append({
    'Model': 'Logistic Regression',
    'Accuracy': lr_accuracy * 100,
    'F1-Score': lr_f1,
    'Precision': lr_precision,
    'Recall': lr_recall,
    'AUC': lr_auc
})

# model 2: random forest
print("\n2. training random forest...")
rf = RandomForestClassifier(labelCol="Delay", featuresCol="features", 
                            numTrees=100, maxDepth=10, maxBins=300, seed=42)
rf_model = rf.fit(train_data)
rf_predictions = rf_model.transform(test_data)

rf_accuracy = evaluator_acc.evaluate(rf_predictions)
rf_f1 = evaluator_f1.evaluate(rf_predictions)
rf_precision = evaluator_precision.evaluate(rf_predictions)
rf_recall = evaluator_recall.evaluate(rf_predictions)
rf_auc = evaluator_auc.evaluate(rf_predictions)

print(f"   accuracy: {rf_accuracy*100:.2f}%")
print(f"   f1-score: {rf_f1:.4f}")
print(f"   precision: {rf_precision:.4f}")
print(f"   recall: {rf_recall:.4f}")
print(f"   auc: {rf_auc:.4f}")

results.append({
    'Model': 'Random Forest',
    'Accuracy': rf_accuracy * 100,
    'F1-Score': rf_f1,
    'Precision': rf_precision,
    'Recall': rf_recall,
    'AUC': rf_auc
})

# feature importance
print("\n   top 10 important features:")
feature_importances = rf_model.featureImportances.toArray()
feature_importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': feature_importances
}).sort_values('Importance', ascending=False)
print(feature_importance_df.head(10).to_string(index=False))

# model 3: decision tree
print("\n3. training decision tree...")
dt = DecisionTreeClassifier(labelCol="Delay", featuresCol="features", maxDepth=15, maxBins=300, seed=42)
dt_model = dt.fit(train_data)
dt_predictions = dt_model.transform(test_data)

dt_accuracy = evaluator_acc.evaluate(dt_predictions)
dt_f1 = evaluator_f1.evaluate(dt_predictions)
dt_precision = evaluator_precision.evaluate(dt_predictions)
dt_recall = evaluator_recall.evaluate(dt_predictions)
dt_auc = evaluator_auc.evaluate(dt_predictions)

print(f"   accuracy: {dt_accuracy*100:.2f}%")
print(f"   f1-score: {dt_f1:.4f}")
print(f"   precision: {dt_precision:.4f}")
print(f"   recall: {dt_recall:.4f}")
print(f"   auc: {dt_auc:.4f}")

results.append({
    'Model': 'Decision Tree',
    'Accuracy': dt_accuracy * 100,
    'F1-Score': dt_f1,
    'Precision': dt_precision,
    'Recall': dt_recall,
    'AUC': dt_auc
})

# model 4: gradient boosted trees
print("\n4. training gradient boosted trees...")
gbt = GBTClassifier(labelCol="Delay", featuresCol="features", maxIter=100, maxDepth=5, maxBins=300, seed=42)
gbt_model = gbt.fit(train_data)
gbt_predictions = gbt_model.transform(test_data)

gbt_accuracy = evaluator_acc.evaluate(gbt_predictions)
gbt_f1 = evaluator_f1.evaluate(gbt_predictions)
gbt_precision = evaluator_precision.evaluate(gbt_predictions)
gbt_recall = evaluator_recall.evaluate(gbt_predictions)
gbt_auc = evaluator_auc.evaluate(gbt_predictions)

print(f"   accuracy: {gbt_accuracy*100:.2f}%")
print(f"   f1-score: {gbt_f1:.4f}")
print(f"   precision: {gbt_precision:.4f}")
print(f"   recall: {gbt_recall:.4f}")
print(f"   auc: {gbt_auc:.4f}")

results.append({
    'Model': 'Gradient Boosted Trees',
    'Accuracy': gbt_accuracy * 100,
    'F1-Score': gbt_f1,
    'Precision': gbt_precision,
    'Recall': gbt_recall,
    'AUC': gbt_auc
})

# compare models
print("\n" + "-" * 60)
print("model comparison")
print("-" * 60)

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))

best_model_idx = results_df['Accuracy'].idxmax()
best_model = results_df.iloc[best_model_idx]
print(f"\nbest model: {best_model['Model']}")
print(f"accuracy: {best_model['Accuracy']:.2f}%")
print(f"f1-score: {best_model['F1-Score']:.4f}")

# confusion matrices
print("\n" + "-" * 60)
print("confusion matrices")
print("-" * 60)

def compute_confusion_matrix(predictions):
    tp = predictions.filter((col("Delay") == 1) & (col("prediction") == 1)).count()
    tn = predictions.filter((col("Delay") == 0) & (col("prediction") == 0)).count()
    fp = predictions.filter((col("Delay") == 0) & (col("prediction") == 1)).count()
    fn = predictions.filter((col("Delay") == 1) & (col("prediction") == 0)).count()
    return np.array([[tn, fp], [fn, tp]])

cm_lr = compute_confusion_matrix(lr_predictions)
cm_rf = compute_confusion_matrix(rf_predictions)
cm_dt = compute_confusion_matrix(dt_predictions)
cm_gbt = compute_confusion_matrix(gbt_predictions)

print("\nlogistic regression:")
print(cm_lr)
print("\nrandom forest:")
print(cm_rf)
print("\ndecision tree:")
print(cm_dt)
print("\ngradient boosted trees:")
print(cm_gbt)

# viz 1: model accuracy
print("creating: ml_1_model_accuracy.png")
plt.figure(figsize=(12, 6))
bars = plt.bar(results_df['Model'], results_df['Accuracy'], color='steelblue', edgecolor='black', linewidth=1.5)
bars[best_model_idx].set_color('#2ecc71')
plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
plt.xlabel('Model', fontsize=14, fontweight='bold')
plt.title('Model Accuracy Comparison', fontsize=18, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right')
for i, v in enumerate(results_df['Accuracy']):
    plt.text(i, v + 0.5, f'{v:.2f}%', ha='center', fontweight='bold', fontsize=11)
plt.ylim([0, max(results_df['Accuracy']) + 5])
plt.grid(True, alpha=0.3, axis='y', linestyle='--')
plt.tight_layout()
plt.savefig('ml_1_model_accuracy.png', dpi=300, bbox_inches='tight')
plt.close()

# viz 2: all metrics
print("creating: ml_2_all_metrics.png")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    if metric == 'Accuracy':
        values = results_df[metric]
    else:
        values = results_df[metric] * 100
    bars = ax.bar(results_df['Model'], values, color='#3498db', edgecolor='black', linewidth=1.2)
    best_idx = values.idxmax()
    bars[best_idx].set_color('#e74c3c')
    ax.set_ylabel(f'{metric} (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'{metric} by Model', fontsize=14, fontweight='bold')
    ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
    for i, v in enumerate(values):
        ax.text(i, v + 1, f'{v:.2f}', ha='center', fontweight='bold', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
plt.tight_layout()
plt.savefig('ml_2_all_metrics.png', dpi=300, bbox_inches='tight')
plt.close()

# viz 3: confusion matrices
print("creating: ml_3_confusion_matrices.png")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
cms = [cm_lr, cm_rf, cm_dt, cm_gbt]
titles = ['Logistic Regression', 'Random Forest', 'Decision Tree', 'Gradient Boosted Trees']
for idx, (cm, title) in enumerate(zip(cms, titles)):
    ax = axes[idx // 2, idx % 2]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=['On-Time', 'Delayed'], yticklabels=['On-Time', 'Delayed'])
    ax.set_title(f'{title}', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('ml_3_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()

# viz 4: feature importance
print("creating: ml_4_feature_importance.png")
plt.figure(figsize=(12, 8))
top_features = feature_importance_df.head(15)
plt.barh(top_features['Feature'], top_features['Importance'], color='#16a085')
plt.xlabel('Importance Score', fontsize=14, fontweight='bold')
plt.ylabel('Feature', fontsize=14, fontweight='bold')
plt.title('Top 15 Most Important Features (Random Forest)', fontsize=18, fontweight='bold', pad=20)
plt.gca().invert_yaxis()
for i, v in enumerate(top_features['Importance']):
    plt.text(v + 0.001, i, f'{v:.4f}', va='center', fontweight='bold', fontsize=10)
plt.tight_layout()
plt.savefig('ml_4_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# viz 5: auc comparison
print("creating: ml_5_auc_comparison.png")
plt.figure(figsize=(10, 6))
bars = plt.bar(results_df['Model'], results_df['AUC'], color='#9b59b6', edgecolor='black', linewidth=1.5)
best_auc_idx = results_df['AUC'].idxmax()
bars[best_auc_idx].set_color('#e67e22')
plt.ylabel('AUC-ROC Score', fontsize=14, fontweight='bold')
plt.xlabel('Model', fontsize=14, fontweight='bold')
plt.title('AUC-ROC Comparison', fontsize=18, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right')
for i, v in enumerate(results_df['AUC']):
    plt.text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold', fontsize=11)
plt.ylim([0, 1.1])
plt.axhline(y=0.5, color='r', linestyle='--', label='Random Classifier', linewidth=2)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3, axis='y', linestyle='--')
plt.tight_layout()
plt.savefig('ml_5_auc_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nall visualizations created successfully!")

results_df.to_csv('ml_results_summary.csv', index=False)
print("saved: ml_results_summary.csv")

feature_importance_df.to_csv('feature_importance.csv', index=False)
print("saved: feature_importance.csv")

# summary
summary = f"""
ml summary

dataset:
- total records: {final_data.count():,}
- training set: {train_count:,} ({train_count/final_data.count()*100:.1f}%)
- testing set: {test_count:,} ({test_count/final_data.count()*100:.1f}%)
- features: {len(feature_columns)}

models trained:
1. logistic regression
2. random forest (100 trees, depth=10)
3. decision tree (depth=15)
4. gradient boosted trees (100 iterations, depth=5)

results:
{results_df.to_string(index=False)}

best model: {best_model['Model']}
- accuracy: {best_model['Accuracy']:.2f}%
- f1-score: {best_model['F1-Score']:.4f}
- precision: {best_model['Precision']:.4f}
- recall: {best_model['Recall']:.4f}
- auc: {best_model['AUC']:.4f}

top 5 features:
{feature_importance_df.head(5).to_string(index=False)}

visualizations:
1. ml_1_model_accuracy.png
2. ml_2_all_metrics.png
3. ml_3_confusion_matrices.png
4. ml_4_feature_importance.png
5. ml_5_auc_comparison.png
"""

print(summary)

with open('ml_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary)
print("\nsaved: ml_summary.txt")

# stop spark
spark.stop()