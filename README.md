# AWS SageMaker
<p align="center">
<img src="https://github.com/ghafeleb/aws-sagemaker/blob/main/images/aws_sagemaker_icon.png" width="50%" alt="AWS SageMaker"/>
  <br>
  <em></em>
</p>


<p align="justify">
This repository is a collection of tutorial steps that showcase my skills and learning journey with AWS SageMaker. AWS SageMaker is a fully managed service that provides every developer and data scientist with the ability to build, train, and deploy machine learning (ML) models quickly.
</p>

# Labelling Data 
To label our image data, we should follow these steps:
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
    <img src="https://github.com/ghafeleb/aws-sagemaker/blob/main/images/labeling.png" width="50%" alt="Labeled data"/>
      <br>
      <em></em>
    </p>
    Sample JSON Lines format output.manifest for a single image:
```
{"source-ref":"s3://****/image_0007.jpeg","vehicle-labeling-demo":3,"vehicle-labeling-demo-metadata":{"class-name":"Helicopter","job-name":"labeling-job/vehicle-labeling-demo","confidence":0.49,"type":"groundtruth/image-classification","human-annotated":"yes","creation-date":"****"}}    
```
