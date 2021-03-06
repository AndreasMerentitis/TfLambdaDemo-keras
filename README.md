# Serverless Machine Learning inference on AWS Lambda with TensorFlow and keras (custom model)

Configured to deploy a TensorFlow model to AWS Lambda using the Serverless framework.
The model is a custom one trained locally with keras. 

by: Mike Moritz

updates by: Andreas Merentitis (ubuntu 20.04, py36)

![relative path 6](/bluriness_pie.png?raw=true "bluriness_pie.png")
![relative path 1](/model_train.png?raw=true "model_train.png")

### Prerequisites

#### Setup serverless

```  
sudo npm install -g serverless

sudo serverless plugin install -n serverless-python-requirements

pip install -r requirements.txt

```
#### Setup AWS credentials

Make sure you have the AWS access key and secrete keys setup locally, following this video [here](https://www.youtube.com/watch?v=KngM5bfpttA)

### Download the code locally

```  
serverless create --template-url https://github.com/AndreasMerentitis/TfLambdaDemo-keras --path tf-lambdakeras
```

### Update S3 bucket to unique name
In serverless.yml:
```  
  environment:
    BUCKET: <your_unique_bucket_name> 
```

### Check the file syntax for any files changed 
```
pyflakes infer.py

```
We can ignore the warning about not using 'unzip_requirements' as its needed to set the requirements for lamda 

### Train the model from scratch

```
source activate py36

python local_train_new_model.py 

tar -zcvf model.tar.gz model_ML.h5
```


### Deploy to the cloud  


```
cd tf-lambdakeras

npm install

sudo serverless deploy --stage dev

aws s3 cp model.tar.gz s3://serverless-ml-1/ --grants read=uri=http://acs.amazonaws.com/groups/global/AllUsers

curl -X POST https://u881f1756g.execute-api.eu-west-1.amazonaws.com/dev/infer -d '{"epoch": "1556995767", "input": {"age": ["34"], "workclass": ["Private"], "fnlwgt": ["357145"], "education": ["Bachelors"], "education_num": ["13"], "marital_status": ["Married-civ-spouse"], "occupation": ["Prof-specialty"], "relationship": ["Wife"], "race": ["White"], "gender": ["Female"], "capital_gain": ["0"], "capital_loss": ["0"], "hours_per_week": ["50"], "native_country": ["United-States"], "income_bracket": [">50K"]}}'

```

### Clean up (remove deployment) 


```
aws s3 rm s3://serverless-ml-1/model.tar.gz

sudo serverless remove --stage dev 
```

# Using data and extending the basic idea from these sources:
* https://github.com/mikepm35/TfLambdaDemo
* https://medium.com/@mike.p.moritz/running-tensorflow-on-aws-lambda-using-serverless-5acf20e00033









