
# coding: utf-8

# # Diabetes prediction using synthesized health records
#
# This notebook explores how to train a machine learning model to predict type 2 diabetes using synthesized patient health records.  The use of synthesized data allows us to learn about building a model without any concern about the privacy issues surrounding the use of real patient health records.
#
# To do
#   None

# Error
#   None

# ## Prerequisites
#
# This project is part of a series of code patterns pertaining to a fictional health care company called Example Health.  This company stores electronic health records in a database on a z/OS server.  Before running the notebook, the synthesized health records must be created and loaded into this database.  Another project, https://github.com/IBM/example-health-synthea, provides the steps for doing this.  The records are created using a tool called Synthea (https://github.com/synthetichealth/synthea), transformed and loaded into the database.

import argparse, datetime, os, pdb, sys
import math, statistics
import numpy as np
import pandas as pd
from pyspark.sql.functions import col
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegressionCV

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils

z_OS_server = False  # Database from z/OS server.
NOTEBOOK = False
DEPLOY = False

EARLY_DEBUGGING = False
DEBUGGING = False
EARLY_TESTING = False
TESTING = True

TO_DO = False

# ## Load and prepare the data

# ### Set up the information needed for a JDBC connection to your database below
# The database must be set up by following the instructions in https://github.com/IBM/example-health-synthea.

# In[1]:
# This is needed when this code deployed in IBM Cloud
credentials_1 = {
  'host':'xxx.yyy.com',
  'port':'nnnn',
  'username':'user',
  'password':'password',
  'database':'location',
  'schema':'SMHEALTH'
}

# ### Define a function to load data from a database table into a Spark dataframe
#
# The partitionColumn, lowerBound, upperBound, and numPartitions options are used to load the data more quickly
# using multiple JDBC connections.  The data is partitioned by patient id.  It is assumed that there are approximately
# 5000 patients in the database.  If there are more or less patients, adjust the upperBound value appropriately.

# Input argument
# Instead of command argument, Jupyter Notebook
given_arg = {
    'in_file': "record.pkl"
}

no_ep = 100
each_b_s = 100
# Dataset will be divided and one group size will be batch size
total_n_group = 10

observations_and_condition_df = pd.read_pickle(given_arg['in_file'])
# observations_and_condition_df = \
#     observations_and_condition_df.set_index('patientid')
oac_original = observations_and_condition_df
merged_observations_df = observations_and_condition_df[[
    'patientid', 'dateofobservation', 'systolic', 'diastolic', 'hdl',
    'ldl', 'bmi']]
# +---------+-----------------+--------+---------+-----+------+-----+
# |patientid|dateofobservation|systolic|diastolic|  hdl|   ldl|  bmi|
# +---------+-----------------+--------+---------+-----+------+-----+
# |        4|       2011-12-17|  105.10|    77.10|71.00| 86.50|57.70|

# observations_and_condition_df.show(5)

# ### Filter the observations for diabetics to remove those taken before diagnosis
#
# This is driven by the way that the diabetes simulation works in Synthea.
# The impact of the condition (diabetes) is not reflected
# in the observations until the patient is diagnosed
# with the condition in a wellness visit.
# Prior to that the patient's observations
# won't be any different from a non-diabetic patient.
# Therefore we want only the observations at the time the patients were diabetic.

# 1st trial
# This generate only index
# observations_and_condition_df = (
#     observations_and_condition_df.filter(
#         (observations_and_condition_df["diabetic"] == 0) |
#         ((observations_and_condition_df["dateofobservation"] >=
#         observations_and_condition_df["start"])))
# )

# 2nd trial - original code
# observations_and_condition_df = (
#     observations_and_condition_df.filter(
#     (col("diabetic") == 0) |
#     ((col("dateofobservation") >= col("start"))))
# )

# 3rd trial
observations_and_condition_df = observations_and_condition_df[
    (observations_and_condition_df["diabetic"] == 0) |
    (observations_and_condition_df["dateofobservation"] >=
    observations_and_condition_df["start"])]

# ### Reduce the observations to a single observation per
# patient (the earliest available observation)
# Sort by patient ID and date of observation
w = pd.DataFrame(observations_and_condition_df.sort_values(
		by = ['patientid','dateofobservation'],
		ascending = [True,True]))

# Keep the earlist date of observation
first_observation_df = pd.DataFrame(columns=w.keys())
for i_dx, i in enumerate(w.iloc):
    if (i_dx == 0):
        first_observation_df = first_observation_df.append(i)
    else:
        if (i['patientid'] != i_pre['patientid']):
            first_observation_df = first_observation_df.append(i)
    i_pre = i

if (EARLY_DEBUGGING):
    pdb.set_trace()

# ## Visualize data
#
# At this point we have collected some observations which might be relevant to making a diabetes prediction.  The next step is to look for relationships between those observations and having diabetes.  There are many tools that help visualize data to look for relationships.  One of the easiest ones to use is called Pixiedust (https://github.com/pixiedust/pixiedust).
#
# Install the pixiedust visualization tool.
# !pip install --upgrade pixiedust

# ### Use Pixiedust to visualize whether observations correlate with diabetes
#
# The PixieDust interactive widget appears when you run this cell.
# * Click the chart button and choose Scatter Plot.
# * Click the chart options button.  Drag "ldl" into the Keys box and drag "hdl" into the Values box.
# Set the # of Rows to Display to 5000.  Click OK to close the chart options.
# * Select bokeh from the Renderer dropdown menu.
# * Select diabetic from the Color dropdown menu.
#
# The scatter plot chart appears.
#
# Click Options and try replacing "ldl" and "hdl" with other attributes.

if (NOTEBOOK):
    import pixiedust

    display(first_observation_df)

# ## Build and train the model
#
# The visualization of the data showed that the strongest predictors of diabetes are the cholesterol observations.  This is an artifact of the diabetes simulation used to create the synthesized data.  The simulation uses a distinct range of HDL readings for diabetic vs. non-diabetic patients.
#
# The simulation increases the chance of high blood pressure (hypertension) for diabetics but the non-diabetic patients also can have high blood pressure.  Therefore the correlation of high blood pressure to diabetes isn't very strong.
#
# The simulation does not change the weight of any diabetic patients so BMI has no correlation.
#
# Let's continue using HDL and systolic blood pressure as the features for the model.  In reality more features would be needed to build a usable model.
#
# Create a pipeline that assembles the feature columns and runs a logistic regression algorithm.  Then use the observation data to train the model.

# vectorAssembler_features = VectorAssembler(inputCols=["hdl", "systolic"],
#     outputCol="features")
#
# lr = LogisticRegression(featuresCol = 'features',
#     labelCol = 'diabetic', maxIter=10)
#
# pipeline = Pipeline(stages=[vectorAssembler_features, lr])

# DataFrame --> Numpy Array
#  hdl  systolic  diabetic
# 0   79       105         1
# 1   79       105         0
# 2   80       105         1
# 3   79       104         1
# 4   79       105         0
#
# array([[  1.,  79., 105.],
#        [  0.,  79., 105.],
#        [  1.,  80., 105.],
#        [  1.,  79., 104.],
#        [  0.,  79., 105.]])

vectorizer = DictVectorizer(sparse=False)
first_observation_dict = \
    first_observation_df[["hdl", "systolic", "diabetic"]].to_dict('records')
x_y = vectorizer.fit_transform(first_observation_dict)

# ### Split the observation data into two portions
#
# The larger portion (80% of the data) is used to train the model.
# The smaller portion (20% of the data) is used to test the model.

# Error spot
# *** AttributeError: 'DataFrame' object has no attribute 'randomSplit'
# split_data = first_observation_df.randomSplit([0.8, 0.2], 24)
# train_data = split_data[0]
# test_data = split_data[1]

# ''' 80% and 20% '''
train_x, test_x, train_y, test_y = train_test_split(x_y[:, 1:], x_y[:, 0],
    train_size=0.8, test_size=0.2, random_state=0)

unique = list(set(train_y))
train_y_number = []
for i in train_y:
    train_y_number.append(unique.index(i))
unique = list(set(test_y))
test_y_number = []
for i in test_y:
    test_y_number.append(unique.index(i))

# (1) Linear regression
lr_1 = LinearRegression()
lr_1.fit(train_x, train_y_number)  # x needs to be 2d for LinearRegression
print ("Linear Regression")
print ("Accuracy = {:.2f}".format(lr_1.score(test_x, test_y_number)))

# (2) Logistic regression
lr = LogisticRegressionCV()
lr.fit(train_x, train_y)

print ("Logistic Regression")
print ("Accuracy = {:.2f}".format(lr.score(test_x, test_y)))

# (3) Neural Network
# ''' Keras with one hidden layer and 16 units '''
def one_hot_encode_object_array(arr):
    uniques, ids = np.unique(arr, return_inverse=True)
    return np_utils.to_categorical(ids, len(uniques))

train_y_ohe = one_hot_encode_object_array(train_y)
test_y_ohe = one_hot_encode_object_array(test_y)

if (DEBUGGING):
    pdb.set_trace()

model = Sequential()
model.add(Dense(16, input_shape=(2,)))
model.add(Activation('sigmoid'))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(optimizer='adam', loss='binary_crossentropy',
    metrics=["accuracy"])

# Error
# sparse_categorical_crossentropy
# tensorflow.python.framework.errors_impl.InvalidArgumentError:  logits and labels must have the same first dimension, got logits shape [393,2] and labels shape [786]
# "sparse_categorical_crossentropy", 'categorical_crossentropy'

each_b_s = int(train_y_ohe.shape[0] / total_n_group)

if (TESTING):
    print ("Dataset size: {}, batch size: {}, total epoch: {}".format(
    train_y_ohe.shape[0], each_b_s, no_ep
    ))
model.fit(train_x, train_y_ohe, epochs=no_ep, batch_size=each_b_s, verbose=0);

loss, accuracy = model.evaluate(test_x, test_y_ohe, verbose=0)
print ("Neural Network")
print ("Accuracy = {:.2f}".format(accuracy))

# pdb.set_trace()

# ### Train the model
# model = pipeline.fit(train_data)

# ## Evaluate the model
#
# One way to evaluate the model is to plot a precision/recall curve.
#
# Precision measures the percentage of the predicted true outcomes that are actually true.
#
# Recall measures the percentage of the actual true conditions that are predicted as true.
#
# Ideally we want both precision and recall to be 100%.
# We want all of the diabetes predictions to actually have diabetes (precision = 1.0).
# We want all of the actual diabetics to be predicted to be diabetic (recall = 1.0).
#
# The model computes the probability of a true condition and then compares that to a threshold
# (by default 0.5) to make a final true of false determination.  The precision/recall curve plots
# precision and recall at various threhold values.

# Plot the model's precision/recall curve.

if (NOTEBOOK):
    get_ipython().run_line_magic('matplotlib', 'inline')

if (TO_DO):
    trainingSummary = model.stages[-1].summary

    pr = trainingSummary.pr.toPandas()
    plt.plot(pr['recall'],pr['precision'])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.show()

# Let's use the model to make predictions using the test data.  We'll leave the threshold for deciding between a true or false result at the default value of 0.5.
# predictions = model.transform(test_data)

x = test_x
prediction = model.predict(
    x, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10,
    workers=1, use_multiprocessing=True
)

if (TESTING):
    pdb.set_trace()

# Compute recall and precision for the test predictions to see how well the model does.
tp, fp, fn = 0, 0, 0
# 1 is diabetic in test label
# Precition probability of non diabetic and probabilit of diabetic
for i_dx, i in enumerate(prediction):
    if (i[0] > i[1]):
        cur_pred = 0
    else:
        cur_pred = 1

    if (cur_pred == 1 and test_y[i_dx]) == 1:
        tp += 1
    elif (cur_pred == 1 and test_y[i_dx]) == 0:
        fp += 1
    elif (cur_pred == 0 and test_y[i_dx]) == 1:
        fn += 1

print ("True positives  = %s" % tp)
print ("False positives = %s" % fp)
print ("False negatives = %s" % fn)

if (tp + fn != 0):
    print ("Recall = %s" % (tp / (tp + fn)))
if (tp + fp != 0):
    print ("Precision = %s" % (tp / (tp + fp)))


# ## Publish and deploy the model
#
# In this section you will learn how to store the model in the Watson Machine Learning repository by using the repository client.
#
# First, please install the client library.
if (DEPLOY):
    get_ipython().system('rm -rf $PIP_BUILD/watson-machine-learning-client')
    get_ipython().system('pip install watson-machine-learning-client --upgrade')


    # ### Enter your Watson Machine Learning service instance credentials here
    # They can be found in the Service Credentials tab of
    # the Watson Machine Learning service instance that you created on IBM Cloud.

    wml_credentials={
      "url": "https://xxx.ibm.com",
      "username": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
      "password": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
      "instance_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
    }

    # ### Publish the model to the repository using the client

    from watson_machine_learning_client import WatsonMachineLearningAPIClient

    client = WatsonMachineLearningAPIClient(wml_credentials)

    model_props = {
        client.repository.ModelMetaNames.NAME: "diabetes-prediction-1",
    }

    stored_model_details = client.repository.store_model(model, meta_props=model_props, training_data=train_data, pipeline=pipeline)

    model_uid            = client.repository.get_model_uid( stored_model_details )
    print( "model_uid: ", model_uid )

    # ### Deploy the model as a web service

    deployment_details = client.deployments.create(model_uid, 'diabetes-prediction-1 deployment')

    scoring_endpoint = client.deployments.get_scoring_url(deployment_details)
    print(scoring_endpoint)

    # ### Call the web service to make a prediction from some sample data

    scoring_payload = {
        "fields": ["hdl", "systolic"],
        "values": [[45.0, 156.6]]
    }

    score = client.deployments.score(scoring_endpoint, scoring_payload)

    print(str(score))
