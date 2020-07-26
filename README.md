# Serverless Machine Learning on AWS Lambda with TensorFlow

Configured to deploy a TensorFlow model to AWS Lambda using the Serverless framework.

by: Mike Moritz

updates by: Andreas Merentitis (ubuntu 20.04, py36)

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
serverless create --template-url https://github.com/AndreasMerentitis/TfLambdaDemo --path tf-lambda
```

### Update S3 bucket to unique name
In serverless.yml:
```  
  environment:
    BUCKET: <your_unique_bucket_name> 
```


### Train the model from scratch

```
source activate py36

python train_new_model.py 

tar -zcvf model.tar.gz model_ML.h5
```


### Deploy to the cloud  

```
cd tf-lambda

npm install

sudo serverless deploy --stage dev

curl -X POST https://u881f1756g.execute-api.eu-west-1.amazonaws.com/dev/infer -d '{"epoch": "1556995767", "input": {"age": ["34"], "workclass": ["Private"], "fnlwgt": ["357145"], "education": ["Bachelors"], "education_num": ["13"], "marital_status": ["Married-civ-spouse"], "occupation": ["Prof-specialty"], "relationship": ["Wife"], "race": ["White"], "gender": ["Female"], "capital_gain": ["0"], "capital_loss": ["0"], "hours_per_week": ["50"], "native_country": ["United-States"], "income_bracket": [">50K"]}}'

sudo serverless remove --stage dev 
```

![relative path 1](/model_train.png?raw=true "model_train.png")


# Using data and extending the basic idea from these sources:
* https://github.com/mikepm35/TfLambdaDemo
* https://medium.com/@mike.p.moritz/running-tensorflow-on-aws-lambda-using-serverless-5acf20e00033









