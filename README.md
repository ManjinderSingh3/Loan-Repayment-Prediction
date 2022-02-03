# Loan Repayment Prediction Model deployed on AWS Sagemaker

# AWS Sagemaker Steps ☁️
1. Open the AWS Sagemaker Service. We have two options to Create,Train and Deploy our Machine Learning/ Deep Learning Model:
- Sagemaker Studio
- Notebook Instance
2. I am creating Notebook Instance for this project. `Amazon Sagemaker -> Notebook Instances -> Create Notebook Instance`
3. Now we need to create S3 bucket to store Train and Test data. Additionally, trained model will also be saved in S3 bucket.
4. Before creating bucket check the region where your Notebook Instance is created from a dropdown near to profile section on top right corner.
5. In my example region is `us-east-2`.
6. Python Script to create S3 bucket
```
import boto3
s3 = boto3.resource('s3')
try:
    if resource_region =='us-east-2':
        s3.create_bucket(Bucket=bucket_name,
                        CreateBucketConfiguration={'LocationConstraint': 'us-east-2'})
    print('Bucket created successfully')
except Exception as e:
    print('S3 Error:',e)
```
7. Upload the Dataset on Notebook Instance. There could be an online link to download data or you can simply upload it from local system.
8. Once you have dataset available on your Notebook Instance we need to upload it to S3 bucket.
```
import os
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(model_name, '<S3-folder-path>')).upload_file('<notebook-instance-dataset-path>')
```
9. Perform all the necessary steps involved in Life Cycle of Data Science Project. In this example I have performed **Exploratory Data Analysis**, 
**Feature Engineering**, **Model Creation**, and **Hyperparameter Tuning**.
10. After doing all the necessary steps in data pre-processing, store the processed data to S3 bucket.
11. Read the final processed data from S3 bucket and split it into Train and Test set.
```
df = pd.read_csv('s3://<file-path-of-dataset-stored-in-S3-bucket>')
train, test = np.split(df.sample(frac=1, random_state=1), [int(0.8 * len(df))])
```
12. Now store this `Train` and `Test` data to S3 bucket with below mention line of code. Sagemaker will fetch these `.csv` files from S3 bucket while training and Testing the model.  

**a. Store Train Data to S3 bucket**
```
train.to_csv('train.csv',index=False) 
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(model_name, '<S3-path-to-store-training-set>')).upload_file('<file-present-on-Notebook-instance>')
```
**b. Store Train Data to S3 bucket**
```
test.to_csv('test.csv',index=False)
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(model_name, '<S3-path-to-store-training-set>')).upload_file('<file-present-on-Notebook-instance>')

```
13. Now Model Creation part comes. AWS Sagemaker provide some pre-built models. We can use their containerized image and train the model. However, we also have an 
option to create our own model.
14. We now need to access training and testing data from S3 bucket. Below mentioned command fetch data.  

**a. Fetching Training Data from S3 bucket**
```
s3_input_train = sagemaker.TrainingInput(s3_data='s3://{}/{}/s3_dataset/train'.format(bucket_name, model_name), content_type='csv')
```
  **b. Fetching Testing Data from S3 bucket**
```
s3_input_test = sagemaker.TrainingInput(s3_data='s3://{}/{}/s3_dataset/test'.format(bucket_name, model_name), content_type='csv')
```
15. The below example is of XGBoost model. I have used pre-built model.
```
from sagemaker.amazon.amazon_estimator import get_image_uri
container = get_image_uri(boto3.Session().region_name,
                          'xgboost', 
                          repo_version='1.0-1')
estimator = sagemaker.estimator.Estimator(image_name=container, 
                                          role=sagemaker.get_execution_role(),
                                          train_instance_count=1, 
                                          train_instance_type='ml.m5.2xlarge', 
                                          train_volume_size=5, # 5 GB 
                                          train_use_spot_instances=True,
                                          train_max_run=300,
                                          train_max_wait=600)  
estimator.fit({'train': s3_input_train,'validation': s3_input_test})
```
15. Once model is create we can deploy the model 


