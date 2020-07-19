# Serverless Machine Learning on AWS Lambda with TensorFlow

Configured to deploy a TensorFlow model to AWS Lambda using the Serverless framework.

by: Mike Moritz

updated by: Andreas Merentitis (Ubuntu 20.04 and py36) 

More info here:  [https://coderecipe.ai/architectures/16924675](https://coderecipe.ai/architectures/16924675)

### Prerequisites

#### Setup serverless

```  
sudo npm install -g serverless

sudo serverless plugin install -n serverless-python-requirements

pip install -r requirements.txt

```
#### Setup AWS credentials

Make sure you have AWS access key and secrete keys setup locally, following this video [here](https://www.youtube.com/watch?v=KngM5bfpttA)

### Download the code locally

```  
sudo serverless create --template-url https://github.com/mikepm35/TfLambdaDemo --path tf-lambda
```

### Update S3 bucket to unique name
In serverless.yml:
```  
  environment:
    BUCKET: <your_unique_bucket_name> 
```

### Deploy to the cloud  

```
cd tf-lambda

npm install

sudo serverless deploy --stage dev
```
