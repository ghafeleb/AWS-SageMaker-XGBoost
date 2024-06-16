# AWS SageMaker
<p align="center">
<img src="https://github.com/ghafeleb/aws-sagemaker/blob/main/images/aws_sagemaker_icon.png" width="75%" alt="AWS SageMaker"/>
  <br>
  <em></em>
</p>



<p align="justify">
This repository is a collection of tutorial steps that showcase my skills and learning journey with AWS SageMaker following <a href="https://aws.amazon.com/sagemaker/getting-started/?refid=ap_card">Amazon SageMaker tutorials</a>. AWS SageMaker is a fully managed service that provides every developer and data scientist with the ability to build, train, and deploy machine learning (ML) models quickly. The prerequisite of all the steps is creating an AWS account.
</p>

## Contents

1. [Labeling Data ](#labeling-data)
2. [Build and Train a Machine Learning Model Locally](#build-and-train-a-machine-learning-model-locally)
   - [Overfitting Analysis](#overfitting-analysis)

---

# Labeling Data 
In this section, we label samples from  using Amazon Mechanical Turk. To label our image data, we should follow these steps:
1. Set up the Amazon SageMaker Studio domain
2. Set up a SageMaker Studio notebook
3. Create the labeling job
   
    3.1. Run the following code in the Jupyter Notebook to download:
    ```
    import sagemaker
    sess = sagemaker.Session
    bucket = sess.default_bucket()
    !aws s3 sync s3://sagemaker-sample-files/datasets/image/caltech-101/inference/ s3://{bucket}/ground-truth-demo/images/
    ```
    3.2. Assign the labeling job to Amazon Mechanical Turk. The result for the sample data is
    <p align="center">
    <img src="https://github.com/ghafeleb/aws-sagemaker/blob/main/images/labeling.png" width="95%" alt="Labeled data"/>
      <br>
      <em></em>
    </p>
    Sample JSON Lines format output.manifest for a single image:
    
    ```
    {"source-ref":"s3://****/image_0007.jpeg","vehicle-labeling-demo":3,"vehicle-labeling-demo-metadata":{"class-name":"Helicopter","job-name":"labeling-job/vehicle-labeling-demo","confidence":0.49,"type":"groundtruth/image-classification","human-annotated":"yes","creation-date":"****"}}    
    ```

#  Build and Train a Machine Learning Model Locally
This section utilizes the XGBoost framework to prototype a binary classification model to predict fraudulent claims on synthetic auto insurance claims dataset. 

## Sample Data Table
We will use a synthetically generated auto insurance claims dataset about claims and customers, along with a fraud column indicating whether a claim was fraudulent or otherwise. 
| Claim ID | Customer ID | Claim Amount | Age  | Vehicle Type | Accident Date | Claim Date  | Fraudulent |
|----------|-------------|--------------|------|--------------|---------------|-------------|------------|
| 001      | 1001        | 5000         | 45   | Car          | 2023-01-01    | 2023-01-10  | No         |
| 002      | 1002        | 7000         | 34   | Truck        | 2023-01-05    | 2023-01-15  | Yes        |
| 003      | 1003        | 3000         | 29   | Car          | 2023-01-03    | 2023-01-12  | No         |
| 004      | 1004        | 10000        | 54   | SUV          | 2023-01-07    | 2023-01-17  | Yes        |
| 005      | 1005        | 2000         | 41   | Car          | 2023-01-02    | 2023-01-11  | No         |

*Note: This is a sample representation of the data. The actual dataset has different values and contains more features and records.*

## Training Model
In this section, we train the XGBoost framework to build a binary classification model on the synthetic dataset to predict the likelihood of a claim being fraudulent. The steps for training our XGBoost model on AWS are as follows:
1. Create a new notebook file on SageMaker Studio. Use <a href="https://docs.aws.amazon.com/sagemaker/latest/dg/notebooks-available-images.html">`Data Science 2.0`</a> image that includes the most commonly used Python packages and libraries.
2. Update aiobotocore and install xgboost package:
```
%pip install --upgrade -q aiobotocore
%pip install -q  xgboost==1.3.1
```
3. Load the necessary packages and the synthetic auto-insurance claim dataset from a public S3 bucket named `sagemaker-sample-files`:
```
import pandas as pd
setattr(pd, "Int64Index", pd.Index)
setattr(pd, "Float64Index", pd.Index)
import boto3
import sagemaker
import json
import joblib
import xgboost as xgb
from sklearn.metrics import roc_auc_score

# Set SageMaker and S3 client variables
sess = sagemaker.Session()

region = sess.boto_region_name
s3_client = boto3.client("s3", region_name=region)

sagemaker_role = sagemaker.get_execution_role()

# Set read and write S3 buckets and locations
write_bucket = sess.default_bucket()
write_prefix = "fraud-detect-demo"

read_bucket = "sagemaker-sample-files"
read_prefix = "datasets/tabular/synthetic_automobile_claims" 

train_data_key = f"{read_prefix}/train.csv"
test_data_key = f"{read_prefix}/test.csv"
model_key = f"{write_prefix}/model"
output_key = f"{write_prefix}/output"

train_data_uri = f"s3://{read_bucket}/{train_data_key}"
test_data_uri = f"s3://{read_bucket}/{test_data_key}"
```
The following two lines were added to prevent pandas raising errors:
```
setattr(pd, "Int64Index", pd.Index)
setattr(pd, "Float64Index", pd.Index)
```
5. Tune the binary XGBoost classification model to classify the _fraud_ column by using a small portion of complete data:
```
hyperparams = {
                "max_depth": 3,
                "eta": 0.2,
                "objective": "binary:logistic",
                "subsample" : 0.8,
                "colsample_bytree" : 0.8,
                "min_child_weight" : 3
              }
num_boost_round = 100
nfold = 3
early_stopping_rounds = 10
# Set up data input
label_col = "fraud"
data = pd.read_csv(train_data_uri)
# Read training data and target
train_features = data.drop(label_col, axis=1)
train_label = pd.DataFrame(data[label_col])
dtrain = xgb.DMatrix(train_features, label=train_label)
# Cross-validate on training data
cv_results = xgb.cv(
    params=hyperparams,
    dtrain=dtrain,
    num_boost_round=num_boost_round,
    nfold=nfold,
    early_stopping_rounds=early_stopping_rounds,
    metrics=["auc"],
    seed=10,
)
metrics_data = {
    "binary_classification_metrics": {
        "validation:auc": {
            "value": cv_results.iloc[-1]["test-auc-mean"],
            "standard_deviation": cv_results.iloc[-1]["test-auc-std"]
        },
        "train:auc": {
            "value": cv_results.iloc[-1]["train-auc-mean"],
            "standard_deviation": cv_results.iloc[-1]["train-auc-std"]
        },
    }
}
print(f"Cross-validated train-auc:{cv_results.iloc[-1]['train-auc-mean']:.2f}")
print(f"Cross-validated validation-auc:{cv_results.iloc[-1]['test-auc-mean']:.2f}")
```
The trained model shows cross-validated train-AUC of 0.9 and cross-validated validation-AUC of 0.78. 
6. After tuning, we train the model with the complete data:
```
data = pd.read_csv(test_data_uri)
test_features = data.drop(label_col, axis=1)
test_label = pd.DataFrame(data[label_col])
dtest = xgb.DMatrix(test_features, label=test_label)

model = (xgb.train(params=hyperparams, dtrain=dtrain, evals = [(dtrain,'train'), (dtest,'eval')], num_boost_round=num_boost_round, 
                  early_stopping_rounds=early_stopping_rounds, verbose_eval = 0)
        )

# Test model performance on train and test sets
test_pred = model.predict(dtest)
train_pred = model.predict(dtrain)

test_auc = roc_auc_score(test_label, test_pred)
train_auc = roc_auc_score(train_label, train_pred)

print(f"Train-auc:{train_auc:.2f}, Test-auc:{test_auc:.2f}")
```
The trained model shows train-AUC=0.95 and test-AUC=0.85. 

7. Finally, save the model and its performance results as JSON files:
```
# Save model and performance metrics locally
with open("./metrics.json", "w") as f:
    json.dump(metrics_data, f)
with open("./xgboost-model", "wb") as f:
    joblib.dump(model, f)        
# Upload model and performance metrics to S3
metrics_location = output_key + "/metrics.json"
model_location = model_key + "/xgboost-model"
s3_client.upload_file(Filename="./metrics.json", Bucket=write_bucket, Key=metrics_location)
s3_client.upload_file(Filename="./xgboost-model", Bucket=write_bucket, Key=model_location)
```


## Overfitting Analysis
By increasing the early_stopping_rounds to 100, our training-AUC improves to 0.99, but the validation-auc drops to 0.74 due to severe overfitting:
    <p align="center">
    <img src="https://github.com/ghafeleb/aws-sagemaker/blob/main/images/xgboost_overfit.png" width="95%" alt="Overfitting in XGBoost"/>
      <br>
      <em></em>
    </p>
    By reducing the ratio of features used (i.e. columns used), we get the optimal validation-AUC 0.79 by reducing the overfitting:
    <p align="center">
    <img src="https://github.com/ghafeleb/aws-sagemaker/blob/main/images/xgboost_improved.png" width="95%" alt="Reduce overfitting in XGBoost"/>
      <br>
      <em></em>
    </p>

# Train, Tune, and Evaluate a Machine Learning Model (XGBoost)
This section trains, tunes, and evaluates a machine learning model using Amazon SageMaker Studio and Amazon SageMaker Clarify. We'll use a synthetic auto insurance claims dataset to build a binary classification model with the XGBoost framework, aimed at predicting fraudulent claims. Additionally, you'll learn how to detect bias in your model and understand its predictions, deploy the model to a real-time inference endpoint, and evaluate its performance through sample predictions and feature impact analysis.
