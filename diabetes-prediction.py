
# coding: utf-8

# # Diabetes prediction using synthesized health records
#
# This notebook explores how to train a machine learning model to predict type 2 diabetes using synthesized patient health records.  The use of synthesized data allows us to learn about building a model without any concern about the privacy issues surrounding the use of real patient health records.
#
# To do
#   Abandon
#   Modify load_data_from_database function
#       Original method not used due to hardware limit

# Error

# ## Prerequisites
#
# This project is part of a series of code patterns pertaining to a fictional health care company called Example Health.  This company stores electronic health records in a database on a z/OS server.  Before running the notebook, the synthesized health records must be created and loaded into this database.  Another project, https://github.com/IBM/example-health-synthea, provides the steps for doing this.  The records are created using a tool called Synthea (https://github.com/synthetichealth/synthea), transformed and loaded into the database.

import datetime, os, pdb, sys
import math, statistics
import numpy as np
import pandas as pd
from pyspark.sql.functions import col
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

z_OS_server = False  # Database from z/OS server.
NOTEBOOK = False

DEBUGGING = True

# ## Load and prepare the data

# ### Set up the information needed for a JDBC connection to your database below
# The database must be set up by following the instructions in https://github.com/IBM/example-health-synthea.

# In[1]:


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

# In[2]:


def load_data_from_database(table_name):
    return (
        spark.read.format("jdbc").options(
            driver = "com.ibm.db2.jcc.DB2Driver",
            url = "jdbc:db2://" + credentials_1["host"] + ":" + credentials_1["port"] + "/" + credentials_1["database"],
            user = credentials_1["username"],
            password = credentials_1["password"],
            dbtable = credentials_1["schema"] + "." + table_name,
            partitionColumn = "patientid",
            lowerBound = 1,
            upperBound = 5000,
            numPartitions = 10
        ).load()
    )



# ### Read patient observations from the database
if (z_OS_server):
    #
    # The observations include things like blood pressure and cholesterol readings which are potential features for our model.

    # In[3]:


    observations_df = load_data_from_database("OBSERVATIONS")

    observations_df.show(5)


    # ### The observations table has a generalized format with a separate row per observation
    #
    # Let's collect the observations that may be of interest in making a diabetes prediction.
    # First, select systolic blood pressure readings from the observations.  These have code 8480-6.

    # In[ ]:


    from pyspark.sql.functions import col

    systolic_observations_df = (
        observations_df.select("patientid", "dateofobservation", "numericvalue")
                       .withColumnRenamed("numericvalue", "systolic")
                       .filter((col("code") == "8480-6"))
      )


    systolic_observations_df.show(5)


    # ### Select other observations of potential interest
    #
    # * Select diastolic blood pressure readings (code 8462-4).
    # * Select HDL cholesterol readings (code 2085-9).
    # * Select LDL cholesterol readings (code 18262-6).
    # * Select BMI (body mass index) readings (code 39156-5).

    # In[ ]:


    diastolic_observations_df = (
        observations_df.select("patientid", "dateofobservation", "numericvalue")
                       .withColumnRenamed('numericvalue', 'diastolic')
                       .filter((col("code") == "8462-4"))
        )

    hdl_observations_df = (
        observations_df.select("patientid", "dateofobservation", "numericvalue")
                       .withColumnRenamed('numericvalue', 'hdl')
                       .filter((col("code") == "2085-9"))
        )

    ldl_observations_df = (
        observations_df.select("patientid", "dateofobservation", "numericvalue")
                       .withColumnRenamed('numericvalue', 'ldl')
                       .filter((col("code") == "18262-6"))
        )

    bmi_observations_df = (
        observations_df.select("patientid", "dateofobservation", "numericvalue")
                       .withColumnRenamed('numericvalue', 'bmi')
                       .filter((col("code") == "39156-5"))
        )


    # ### Join the observations for each patient by date into one dataframe

    # In[ ]:


    merged_observations_df = (
        systolic_observations_df.join(diastolic_observations_df, ["patientid", "dateofobservation"])
                                .join(hdl_observations_df, ["patientid", "dateofobservation"])
                                .join(ldl_observations_df, ["patientid", "dateofobservation"])
                                .join(bmi_observations_df, ["patientid", "dateofobservation"])
    )

    merged_observations_df.show(5)
    # +---------+-----------------+--------+---------+-----+------+-----+
    # |patientid|dateofobservation|systolic|diastolic|  hdl|   ldl|  bmi|
    # +---------+-----------------+--------+---------+-----+------+-----+
    # |        4|       2011-12-17|  105.10|    77.10|71.00| 86.50|57.70|

    # ### Another possible feature is the patient's age at the time of observation
    #
    # Load the patients' birth dates from the database into a dataframe.

    # In[ ]:


    patients_df = load_data_from_database("PATIENT").select("patientid", "dateofbirth")

    patients_df.show(5)


    # Add a column containing the patient's age to the merged observations.

    # In[ ]:


    from pyspark.sql.functions import datediff

    merged_observations_with_age_df = (
      merged_observations_df.join(patients_df, "patientid")
                            .withColumn("age", datediff(col("dateofobservation"), col("dateofbirth"))/365)
                            .drop("dateofbirth")
      )

    merged_observations_with_age_df.show(5)


    # ### Find the patients that have been diagnosed with type 2 diabetes
    #
    # The conditions table contains the conditions that patients have and the date they were diagnosed.
    # Load the patient conditions table and select the patients that have been diagnosed with type 2 diabetes.
    # Keep the date they were diagnosed ("start" column).

    # In[ ]:


    diabetics_df = (
        load_data_from_database("CONDITIONS")
        .select("patientid", "start")
        .filter(col("description") == "Diabetes")
    )

    diabetics_df.show(5)


    # ### Create a "diabetic" column which is the "label" for the model to predict
    #
    # Join the merged observations with the diabetic patients.
    # This is a left join so that we keep all observations for both diabetic and non-diabetic patients.
    # Create a new column with a binary value, 1=diabetic, 0=non-diabetic.
    # This will be the label for the model (the value it is trying to predict).

    # In[ ]:


    from pyspark.sql.functions import when

    observations_and_condition_df = (
        merged_observations_with_age_df.join(diabetics_df, "patientid", "left_outer")
        .withColumn("diabetic", when(col("start").isNotNull(), 1).otherwise(0))
    )

# Number of patients, and there would be multiple observation per patient
total_patient = 5000
no_observation = 4
upperBound =  total_patient * no_observation

# To do
#   Build DataFrame with
#   patientid|d"ateofobservation|systolic|diastolic|  hdl|  ldl|  bmi| age|start|diabetic
# ex)
# +---------+-----------------+--------+---------+-----+-----+-----+-----------------+-----+--------+
# |patientid|dateofobservation|systolic|diastolic|  hdl|  ldl|  bmi|              age|start|diabetic|
# +---------+-----------------+--------+---------+-----+-----+-----+-----------------+-----+--------+
# |      463|       2013-01-26|  113.40|    77.50|77.30|91.40|35.80|52.52876712328767| null|       0|
# |      463|       2010-01-09|  113.50|    70.60|71.20|76.00|35.80|49.47945205479452| null|       0|

temp = ["patientid", "dateofobservation", "systolic", "diastolic", "hdl", "ldl", "bmi",
    "age", "start", "diabetic"]
observations_and_condition_df = pd.DataFrame(columns=temp)

id_start = 1
id_end = id_start + total_patient
observations_and_condition_df["patientid"] = \
    np.random.uniform(low=id_start, high=id_end,
    size=upperBound).astype(np.int)
# systolic 90 ~ 120
# diastolic 60 ~ 80

# To do
# Beta distribution, but what value would be upper limit?
# hdl 60 milligrams per deciliter (mg/dL) of blood or higher

# ldl
# Less than 100 mg/dL	Optimal
# 100-129 mg/dL	Near optimal/above optimal
# 130-159 mg/dL	Borderline high
# 160-189 mg/dL	High
# bmi
# Below 18.5	Underweight
# 18.5 – 24.9	Normal or Healthy Weight
# 25.0 – 29.9	Overweight
# 30.0 and Above	Obese
mean = (90 + 120) / 2
observations_and_condition_df['systolic'] = np.random.normal(
    mean, 0.33, size=upperBound).astype(np.int)
mean = (60 + 80) / 2

observations_and_condition_df['diastolic'] = np.random.normal(
mean, 0.33, size=upperBound).astype(np.int)

limit = 100
observations_and_condition_df['ldl'] = np.random.poisson(
    limit, upperBound)
mean = (18.5 + 24.9) / 2
observations_and_condition_df['bmi'] = np.random.normal(
    mean, 0.33, size=upperBound).astype(np.int)

# Age: Share on Pinterest The average age of onset for
# type 2 diabetes is 45 years
data_of_birth_early = "1958-11-29"
data_of_birth_late = "2017-07-04"
mean = 45
observations_and_condition_df['age'] = np.random.normal(
    mean, 0.33, size=upperBound).astype(np.int)

# Start: Age at the time of diagnosis
# In 2015, adults aged 45 to 64 were the most diagnosed age group for diabetes
start_early = datetime.datetime(1994, 12, 28)
start_late = datetime.datetime(2012, 7, 20)
observations_and_condition_df["start"] = \
    [i.strftime("%Y-%m-%d") for i in
    start_early + (start_late - start_early) * \
    np.random.uniform(low=0, high=1, size=upperBound)]


# Date of observation
observation_early = datetime.datetime(2009, 5, 16)
observation_late = datetime.datetime(2019, 3, 2)
observations_and_condition_df["dateofobservation"] = \
    [i.strftime("%Y-%m-%d") for i in
    observation_early + (observation_late - observation_early) * \
    np.random.uniform(low=0, high=1, size=upperBound)]

# 0: no and 1: diabetic
observations_and_condition_df["diabetic"] = \
    [round(i) for i in np.random.uniform(low=0, high=1,
    size=upperBound)]
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

# In[ ]:

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
oac_original = observations_and_condition_df

observations_and_condition_df = \
    observations_and_condition_df[
    (observations_and_condition_df["diabetic"] == 0) |
    (observations_and_condition_df["dateofobservation"] >=
    observations_and_condition_df["start"])]

if (DEBUGGING):
    pdb.set_trace()

# ### Reduce the observations to a single observation per
# patient (the earliest available observation)

# In[ ]:


# Error spot
w = Window.partitionBy(observations_and_condition_df["patientid"]).orderBy(
    merged_observations_df["dateofobservation"].asc())

first_observation_df = observations_and_condition_df.withColumn(
    "rn", row_number().over(w)).where(col("rn") == 1).drop("rn")


# ## Visualize data
#
# At this point we have collected some observations which might be relevant to making a diabetes prediction.  The next step is to look for relationships between those observations and having diabetes.  There are many tools that help visualize data to look for relationships.  One of the easiest ones to use is called Pixiedust (https://github.com/pixiedust/pixiedust).
#
# Install the pixiedust visualization tool.

# In[ ]:


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

# In[ ]:

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

# In[ ]:


from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

vectorAssembler_features = VectorAssembler(inputCols=["hdl", "systolic"], outputCol="features")

lr = LogisticRegression(featuresCol = 'features', labelCol = 'diabetic', maxIter=10)

pipeline = Pipeline(stages=[vectorAssembler_features, lr])


# ### Split the observation data into two portions
#
# The larger portion (80% of the data) is used to train the model.
# The smaller portion (20% of the data) is used to test the model.

# In[ ]:


split_data = first_observation_df.randomSplit([0.8, 0.2], 24)
train_data = split_data[0]
test_data = split_data[1]


# ### Train the model

# In[ ]:


model = pipeline.fit(train_data)


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

# In[ ]:


# Plot the model's precision/recall curve.

if (NOTEBOOK):
    get_ipython().run_line_magic('matplotlib', 'inline')


trainingSummary = model.stages[-1].summary

pr = trainingSummary.pr.toPandas()
plt.plot(pr['recall'],pr['precision'])
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.show()


# Let's use the model to make predictions using the test data.  We'll leave the threshold for deciding between a true or false result at the default value of 0.5.

# In[ ]:


predictions = model.transform(test_data)


# Compute recall and precision for the test predictions to see how well the model does.

# In[ ]:


pred_and_label = predictions.select("prediction", "diabetic").toPandas()

tp = pred_and_label[(pred_and_label.prediction == 1) & (pred_and_label.diabetic == 1)].count().tolist()[1]
fp = pred_and_label[(pred_and_label.prediction == 1) & (pred_and_label.diabetic == 0)].count().tolist()[1]
fn = pred_and_label[(pred_and_label.prediction == 0) & (pred_and_label.diabetic == 1)].count().tolist()[1]

print("True positives  = %s" % tp)
print("False positives = %s" % fp)
print("False negatives = %s" % fn)

print("Recall = %s" % (tp / (tp + fn)))
print("Precision = %s" % (tp / (tp + fp)))

if (DEPLOY):
    # ## Publish and deploy the model
    #
    # In this section you will learn how to store the model in the Watson Machine Learning repository by using the repository client.
    #
    # First install the client library.

    # In[ ]:


    get_ipython().system('rm -rf $PIP_BUILD/watson-machine-learning-client')
    get_ipython().system('pip install watson-machine-learning-client --upgrade')


    # ### Enter your Watson Machine Learning service instance credentials here
    # They can be found in the Service Credentials tab of the Watson Machine Learning service instance that you created on IBM Cloud.

    # In[ ]:


    wml_credentials={
      "url": "https://xxx.ibm.com",
      "username": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
      "password": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
      "instance_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
    }


    # ### Publish the model to the repository using the client

    # In[ ]:


    from watson_machine_learning_client import WatsonMachineLearningAPIClient

    client = WatsonMachineLearningAPIClient(wml_credentials)

    model_props = {
        client.repository.ModelMetaNames.NAME: "diabetes-prediction-1",
    }

    stored_model_details = client.repository.store_model(model, meta_props=model_props, training_data=train_data, pipeline=pipeline)

    model_uid            = client.repository.get_model_uid( stored_model_details )
    print( "model_uid: ", model_uid )


    # ### Deploy the model as a web service

    # In[ ]:


    deployment_details = client.deployments.create(model_uid, 'diabetes-prediction-1 deployment')

    scoring_endpoint = client.deployments.get_scoring_url(deployment_details)
    print(scoring_endpoint)


    # ### Call the web service to make a prediction from some sample data

    # In[ ]:


    scoring_payload = {
        "fields": ["hdl", "systolic"],
        "values": [[45.0, 156.6]]
    }

    score = client.deployments.score(scoring_endpoint, scoring_payload)

    print(str(score))
