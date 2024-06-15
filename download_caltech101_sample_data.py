import sagemaker

sess = sagemaker.Session()
bucket = sess.default_bucket()

!aws s3 sync s3://sagemaker-sample-files/datasets/image/caltech-101/inference/ s3://{bucket}/ground-truth-demo/images/

print('Copy and paste the below link into a web browser to confirm the ten images were successfully uploaded to your bucket:')
print(f'https://s3.console.aws.amazon.com/s3/buckets/{bucket}/ground-truth-demo/images/')

print('\nWhen prompted by Sagemaker to enter the S3 location for input datasets, you can paste in the below S3 URL')

print(f's3://{bucket}/ground-truth-demo/images/')

print('\nWhen prompted by Sagemaker to Specify a new location, you can paste in the below S3 URL')

print(f's3://{bucket}/ground-truth-demo/labeled-data/')
