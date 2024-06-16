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
3. [Train, Tune, and Evaluate a Machine Learning Model (XGBoost)](#train,-tune,-and-evaluate-a-machine-learning-model-(xgboost))
   - [Script Mode Hyperparameter Tuning of the SageMaker Estimator](#script-mode-hyperparameter-tuning-of-the-sagemaker-estimator)
   - [SageMaker Clarify: Check the Biases of the Model Explain the Model Predictions](#sagemaker-clarify-check-the-biases-of-the-model-explain-the-model-predictions)
   - [Deploy the Model to a Real-time Inference Endpoint](#deploy-the-model-to-a-real-time-inference-endpoint)

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
This section trains, tunes, and evaluates a machine learning model using Amazon SageMaker Studio and Amazon SageMaker Clarify. We'll use a synthetic auto insurance claims dataset to build a binary classification model with the XGBoost framework, aimed at predicting fraudulent claims. Additionally, you'll learn how to detect bias in your model and understand its predictions, deploy the model to a real-time inference endpoint, and evaluate its performance through sample predictions and feature impact analysis. Steps:
1. Open the SageMaker Studio. The instance type of ml.t3.medium would suffice for our purpose. Load Data Science 3.0 image and use Python 3 kernel.
2. Run the following command to make sure you are using the current version of the SageMaker:
```
%pip install sagemaker --upgrade --quiet 
```
3. install the dependencies: 
```
%pip install -q  xgboost==1.3.1 pandas==1.0.5
```
4. Load the packages, including the XGBoost from the built-in framework with a Docker container image:
```
import pandas as pd
import boto3
import sagemaker
import json
import joblib
from sagemaker.xgboost.estimator import XGBoost
from sagemaker.tuner import (
    IntegerParameter,
    ContinuousParameter,
    HyperparameterTuner
)
from sagemaker.inputs import TrainingInput
from sagemaker.image_uris import retrieve
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import CSVDeserializer
```
5. Define and set the SageMaker variables and S3 locations, including the location to read the necessary files and the location for writing the output:
```
# Setting SageMaker variables
sess = sagemaker.Session()
write_bucket = sess.default_bucket()
write_prefix = "fraud-detect-demo"

region = sess.boto_region_name
s3_client = boto3.client("s3", region_name=region)

sagemaker_role = sagemaker.get_execution_role()
sagemaker_client = boto3.client("sagemaker")
read_bucket = "sagemaker-sample-files"
read_prefix = "datasets/tabular/synthetic_automobile_claims" 


# Setting S3 location for read and write operations
train_data_key = f"{read_prefix}/train.csv"
test_data_key = f"{read_prefix}/test.csv"
validation_data_key = f"{read_prefix}/validation.csv"
model_key = f"{write_prefix}/model"
output_key = f"{write_prefix}/output"


train_data_uri = f"s3://{read_bucket}/{train_data_key}"
test_data_uri = f"s3://{read_bucket}/{test_data_key}"
validation_data_uri = f"s3://{read_bucket}/{validation_data_key}"
model_uri = f"s3://{write_bucket}/{model_key}"
output_uri = f"s3://{write_bucket}/{output_key}"
estimator_output_uri = f"s3://{write_bucket}/{write_prefix}/training_jobs"
bias_report_output_uri = f"s3://{write_bucket}/{write_prefix}/clarify-output/bias"
explainability_report_output_uri = f"s3://{write_bucket}/{write_prefix}/clarify-output/explainability"
```
6. Run the following command to define the name of the model and the counts and configurations of the training and inference: 
```
tuning_job_name_prefix = "xgbtune" 
training_job_name_prefix = "xgbtrain"

xgb_model_name = "fraud-detect-xgb-model"
endpoint_name_prefix = "xgb-fraud-model-dev"
train_instance_count = 1
train_instance_type = "ml.m5.xlarge"
predictor_instance_count = 1
predictor_instance_type = "ml.m5.xlarge"
clarify_instance_count = 1
clarify_instance_type = "ml.m5.xlarge"
```
Note: You may need to request an increase in your quota by submitting request on <a href="https://docs.aws.amazon.com/servicequotas/latest/userguide/request-quota-increase.html"></a> for `ml.m5.xlarge for processing job usage`
## Script Mode Hyperparameter Tuning of the SageMaker Estimator
7. Define the training process including parameter definition, model creation, training, and performance evaluation as the following script:
```
%%writefile xgboost_train.py

import argparse
import os
import joblib
import json
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Hyperparameters and algorithm parameters are described here
    parser.add_argument("--num_round", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--objective", type=str, default="binary:logistic")
    parser.add_argument("--eval_metric", type=str, default="auc")
    parser.add_argument("--nfold", type=int, default=3)
    parser.add_argument("--early_stopping_rounds", type=int, default=3)
    

    # SageMaker specific arguments. Defaults are set in the environment variables
    # Location of input training data
    parser.add_argument("--train_data_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    # Location of input validation data
    parser.add_argument("--validation_data_dir", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))
    # Location where trained model will be stored. Default set by SageMaker, /opt/ml/model
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    # Location where model artifacts will be stored. Default set by SageMaker, /opt/ml/output/data
    parser.add_argument("--output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR"))
    
    args = parser.parse_args()

    data_train = pd.read_csv(f"{args.train_data_dir}/train.csv")
    train = data_train.drop("fraud", axis=1)
    label_train = pd.DataFrame(data_train["fraud"])
    dtrain = xgb.DMatrix(train, label=label_train)
    
    
    data_validation = pd.read_csv(f"{args.validation_data_dir}/validation.csv")
    validation = data_validation.drop("fraud", axis=1)
    label_validation = pd.DataFrame(data_validation["fraud"])
    dvalidation = xgb.DMatrix(validation, label=label_validation)

    params = {"max_depth": args.max_depth,
              "eta": args.eta,
              "objective": args.objective,
              "subsample" : args.subsample,
              "colsample_bytree":args.colsample_bytree
             }
    
    num_boost_round = args.num_round
    nfold = args.nfold
    early_stopping_rounds = args.early_stopping_rounds
    
    # Define the grid search combinations
    cv_results = xgb.cv(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        nfold=nfold,
        early_stopping_rounds=early_stopping_rounds,
        metrics=["auc"],
        seed=42,
    )
    
    # Define the model to be trained
    model = xgb.train(params=params, dtrain=dtrain, num_boost_round=len(cv_results))
    
    # Train and validate the model
    train_pred = model.predict(dtrain)
    validation_pred = model.predict(dvalidation)
    
    # Get the train and validation AUC 
    train_auc = roc_auc_score(label_train, train_pred)
    validation_auc = roc_auc_score(label_validation, validation_pred)
    
    print(f"[0]#011train-auc:{train_auc:.2f}")
    print(f"[0]#011validation-auc:{validation_auc:.2f}")

    metrics_data = {"hyperparameters" : params,
                    "binary_classification_metrics": {"validation:auc": {"value": validation_auc},
                                                      "train:auc": {"value": train_auc}
                                                     }
                   }
              
    # Save the evaluation metrics to the location specified by output_data_dir
    metrics_location = args.output_data_dir + "/metrics.json"
    
    # Save the model to the location specified by model_dir
    model_location = args.model_dir + "/xgboost-model"

    with open(metrics_location, "w") as f:
        json.dump(metrics_data, f)

    with open(model_location, "wb") as f:
        joblib.dump(model, f)
```
7. Instantiate the SageMaker esimator (XGBoost estimatir here):
```
# SageMaker estimator

# Set static hyperparameters that will not be tuned
static_hyperparams = {  
                        "eval_metric" : "auc",
                        "objective": "binary:logistic",
                        "num_round": "5"
                      }

xgb_estimator = XGBoost(
                        entry_point="xgboost_train.py",
                        output_path=estimator_output_uri,
                        code_location=estimator_output_uri,
                        hyperparameters=static_hyperparams,
                        role=sagemaker_role,
                        instance_count=train_instance_count,
                        instance_type=train_instance_type,
                        framework_version="1.3-1",
                        base_job_name=training_job_name_prefix
                    )
```
8. Tune the hyperparameters at scale:
```
# Setting ranges of hyperparameters to be tuned
hyperparameter_ranges = {
    "eta": ContinuousParameter(0, 1),
    "subsample": ContinuousParameter(0.7, 0.95),
    "colsample_bytree": ContinuousParameter(0.7, 0.95),
    "max_depth": IntegerParameter(1, 5)
}
```
| Parameter         | Description                                                                                                  |
|-------------------|--------------------------------------------------------------------------------------------------------------|
| `eta`             | Step size shrinkage to prevent overfitting. Shrinks feature weights to make boosting more conservative.      |
| `subsample`       | Ratio of training instances sampled. A value of 0.5 means half of the data is randomly sampled in each iteration, reducing overfitting. |
| `colsample_bytree`| Fraction of features used per tree. Using a subset of features adds randomness and improves generalizability. |
| `max_depth`       | Maximum depth of a tree. Higher values increase model complexity and the risk of overfitting.                 |

9. Set the hyperparameter tuner with a random search process and AUC as the performance measure:
```
objective_metric_name = "validation:auc"

# Setting up tuner object
tuner_config_dict = {
                     "estimator" : xgb_estimator,
                     "max_jobs" : 5,
                     "max_parallel_jobs" : 2,
                     "objective_metric_name" : objective_metric_name,
                     "hyperparameter_ranges" : hyperparameter_ranges,
                     "base_tuning_job_name" : tuning_job_name_prefix,
                     "strategy" : "Random"
                    }
tuner = HyperparameterTuner(**tuner_config_dict)
```
10. Tune the hyperparameters by fitting the model:
```
# Setting the input channels for tuning job
s3_input_train = TrainingInput(s3_data="s3://{}/{}".format(read_bucket, train_data_key), content_type="csv", s3_data_type="S3Prefix")
s3_input_validation = (TrainingInput(s3_data="s3://{}/{}".format(read_bucket, validation_data_key), 
                                    content_type="csv", s3_data_type="S3Prefix")
                      )

tuner.fit(inputs={"train": s3_input_train, "validation": s3_input_validation}, include_cls_metadata=False)
tuner.wait()
```
We can check the Hyperparameter tuning jobs subsection at the AWS SageMaker console to see the info on tuning jobs:
<p align="center">
<img src="https://github.com/ghafeleb/aws-sagemaker/blob/main/images/tuning_jobs.png" width="85%" alt="Hyperparameter tuning jobs"/>
  <br>
  <em></em>
</p>

11. Run the following command to check the tuning summary:
```
# Summary of tuning results ordered in descending order of performance
df_tuner = sagemaker.HyperparameterTuningJobAnalytics(tuner.latest_tuning_job.job_name).dataframe()
df_tuner = df_tuner[df_tuner["FinalObjectiveValue"]>-float('inf')].sort_values("FinalObjectiveValue", ascending=False)
df_tuner
```
Based on the summary of results, the second set of parameters outperforms others and shows AUC of 0.82:
<p align="center">
<img src="https://github.com/ghafeleb/aws-sagemaker/blob/main/images/xgboost_tuning_summary.png" width="95%" alt="Tuning"/>
  <br>
  <em></em>
</p>

## SageMaker Clarify: Check the Biases of the Model Explain the Model Predictions
12. We use SageMaker Clarify to find biases based on feature attribution method. Create a duplicate of the best model based on the tuning results:
```
tuner_job_info = sagemaker_client.describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=tuner.latest_tuning_job.job_name)

model_matches = sagemaker_client.list_models(NameContains=xgb_model_name)["Models"]

if not model_matches:
    _ = sess.create_model_from_job(
            name=xgb_model_name,
            training_job_name=tuner_job_info['BestTrainingJob']["TrainingJobName"],
            role=sagemaker_role,
            image_uri=tuner_job_info['TrainingJobDefinition']["AlgorithmSpecification"]["TrainingImage"]
            )
else:

    print(f"Model {xgb_model_name} already exists.")
```
13. Define the configs for the SageMaker Clarify to check whether there is a bias based on the gender toward female users in the data:
```
train_df = pd.read_csv(train_data_uri)
train_df_cols = train_df.columns.to_list()

clarify_processor = sagemaker.clarify.SageMakerClarifyProcessor(
    role=sagemaker_role,
    instance_count=clarify_instance_count,
    instance_type=clarify_instance_type,
    sagemaker_session=sess,
)

# Data config
bias_data_config = sagemaker.clarify.DataConfig(
    s3_data_input_path=train_data_uri,
    s3_output_path=bias_report_output_uri,
    label="fraud",
    headers=train_df_cols,
    dataset_type="text/csv",
)

# Model config
model_config = sagemaker.clarify.ModelConfig(
    model_name=xgb_model_name,
    instance_type=train_instance_type,
    instance_count=1,
    accept_type="text/csv",
)

# Model predictions config to get binary labels from probabilities
predictions_config = sagemaker.clarify.ModelPredictedLabelConfig(probability_threshold=0.5)

# Bias config
bias_config = sagemaker.clarify.BiasConfig(
    label_values_or_threshold=[0],
    facet_name="customer_gender_female",
    facet_values_or_threshold=[1],
)
```

15. To check the pre-existing bias, we check the Class Imbalance (CI). To check the posttraining bias statistics, we use Difference in Positive Proportions in Predicted Labels (DPPL).
```
clarify_processor.run_bias(
    data_config=bias_data_config,
    bias_config=bias_config,
    model_config=model_config,
    model_predicted_label_config=predictions_config,
    pre_training_methods=["CI"],
    post_training_methods=["DPPL"]
    )

clarify_bias_job_name = clarify_processor.latest_job.name
```
16. Download the SageMaker Clarify results in a PDF format from Amazon S3 Bucket to your local directory in SageMaker Studio:
```
clarify_processor.run_bias(
    data_config=bias_data_config,
    bias_config=bias_config,
    model_config=model_config,
    model_predicted_label_config=predictions_config,
    pre_training_methods=["CI"],
    post_training_methods=["DPPL"]
    )

clarify_bias_job_name = clarify_processor.latest_job.name
```

17. As the [report](3_train_an_ml_model/clarify_bias_output.pdf) shows, there is a pre-existing class imbalance w.r.t. the gender in our data such that females are proportionally less than men:
<p align="center">
<img src="https://github.com/ghafeleb/aws-sagemaker/blob/main/images/class_imbalance.png" width="50%" alt="Tuning"/>
  <br>
  <em></em>
</p>

18. We can also check the feature attribution which quantifies the amount of the effect of the each feature on the final prediction. We use SHAP values to compute the contribution of features to the outcome:

```
explainability_data_config = sagemaker.clarify.DataConfig(
    s3_data_input_path=train_data_uri,
    s3_output_path=explainability_report_output_uri,
    label="fraud",
    headers=train_df_cols,
    dataset_type="text/csv",
)

# Use mean of train dataset as baseline data point
shap_baseline = [list(train_df.drop(["fraud"], axis=1).mean())]

shap_config = sagemaker.clarify.SHAPConfig(
    baseline=shap_baseline,
    num_samples=500,
    agg_method="mean_abs",
    save_local_shap_values=True,
)

clarify_processor.run_explainability(
    data_config=explainability_data_config,
    model_config=model_config,
    explainability_config=shap_config
)
```

19. You can save the output by running:
```
# Copy explainability report and view
!aws s3 cp s3://{write_bucket}/{write_prefix}/clarify-output/explainability/report.pdf ./clarify_explainability_output.pdf
```
The explainability analysis report is provided in [clarify_explanability_output.pdf](3_train_an_ml_model/clarify_explanability_output.pdf). Based on this report, the feature with the most contribution to our output is customer_gender_male:
<p align="center">
<img src="https://github.com/ghafeleb/aws-sagemaker/blob/main/images/global_shap.png" width="85%" alt="clarify_explanability_output"/>
  <br>
  <em></em>
</p>

<p align="center">
<img src="https://github.com/ghafeleb/aws-sagemaker/blob/main/images/local_shap.png" width="85%" alt="clarify_explanability_output2"/>
  <br>
  <em></em>
</p>

We can also check the results of Clarify for bias analysis and explainability analysis under the Experiments section of the SageMaker Studio:
<p align="center">
<img src="https://github.com/ghafeleb/aws-sagemaker/blob/main/images/bias.png" width="95%" alt="clarify_bias_plot"/>
  <br>
  <em></em>
</p>
<p align="center">
<img src="https://github.com/ghafeleb/aws-sagemaker/blob/main/images/explainability.png" width="95%" alt="clarify_explanability_plot"/>
  <br>
  <em></em>
</p>

20. We can also check the contribution of each feature on the prediction output for every individual sample. We check test sample 100:
<p align="center">
<img src="https://github.com/ghafeleb/aws-sagemaker/blob/main/images/local_explanation_sample100.png" width="95%" alt="local_explanation_sample100"/>
  <br>
  <em></em>
</p>
For this customer, the incident day has the maximum contribution.

## Deploy the model to a real-time inference endpoint

21. We want to utilize the best model that was selected from our tuning and use it at the real-time inference endpoint. For this purpose, we use SageMaker SDK.

22. Now, we can deploy the model. For instance:
```
# Sample test data
for i in range(3):
    test_df = pd.read_csv(test_data_uri)
    payload = test_df.drop(["fraud"], axis=1).iloc[i].to_list()
    print(f"Model predicted score : {float(predictor.predict(payload)[0][0]):.3f}, True label : {test_df['fraud'].iloc[0]}")
```
The results are:
<p align="center">
<img src="https://github.com/ghafeleb/aws-sagemaker/blob/main/images/deployed_model.png" width="95%" alt="deployed_model"/>
  <br>
  <em></em>
</p>
The scores are very close to the true label, which shows the power of our model incorrect prediction.
