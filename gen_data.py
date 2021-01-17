# gen_data.py - Generate data
# Cloud Cho,  November 27, 2020


# To do
#   Is Drop function in Pandas keep Index?


# Reference:
#   Data: https://www.kaggle.com/cdc/national-health-and-nutrition-examination-survey
#   Data processing: https://www.kaggle.com/what0919/diabetes-prediction

import argparse, glob
import os, pdb, sys
import datetime
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

RENAME = False

def getting_arg():
    parser = argparse.ArgumentParser(description='Make data')
    parser.add_argument('--in_file', dest='in_file',
                    help='input file name')
    parser.add_argument('--in_folder', dest='in_folder',
                help='input folder name')
    parser.add_argument('--out_file', dest='out_file',
                        help='output file name')
    parser.add_argument('--choice', dest='choice', type=int,
                    help='data type choice')

    return parser

def synthesized(out_file):
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

    # Guide for patient record

    # systolic 90 ~ 120

    # diastolic 60 ~ 80

    # To do
    # Beta distribution, but what value would be upper limit?
    # hdl 60 milligrams per deciliter (mg/dL) of blood or higher
    # Oddly enough, people who naturally have extremely high HDL levels
    # — above 100 mg/dL (2.5mmol/L) — appear to be at higher risk of
    # heart disease. This may be caused by genetic factors.
    # so 60 ~ 100 used

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

    mean = (60 + 100) / 2
    observations_and_condition_df['hdl'] = np.random.normal(
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

    try:
        observations_and_condition_df.to_pickle(out_file)
    except Exception as e:
        print (e.args)
        sys.exit(1)
    else:
        print ("{} saved.".format(out_file))


# Dataset Merge & select attribute
def nhanes(input_files, out_file):

    for i_dx, i in enumerate(input_files):
        try:
            df = pd.read_csv(i)
        except Exception as e:
            print (e.args)
            try:
                df = pd.read_csv(i, encoding= 'unicode_escape')
            except Exception as e:
                print (e.args)
                sys.exit(1)
        df.set_index('SEQN', inplace=True)  # SEQN is ID.
        if (i_dx == 0):
            df_all = df
        else:
            df_all = df_all.join(df, how="outer")  # Combine by ID

    #sel = VarianceThreshold(threshold=(.8 * (1 - .8)))

    #sel.fit_transform(df)

    # df_all.describe()

    # NA handling, Feature selection
    # df.dropna(axis=1, how='all')
    # df.dropna(axis=0, how='all')

    if (RENAME):
        df_all = df_all.rename(columns = {'SEQN' : 'ID',
                              'RIAGENDR' : 'Gender',
                              'DMDYRSUS' : 'Years_in_US', # Nan -> american i guess
                              'INDFMPIR' : 'Family_income',
                              'LBXGH' : 'GlycoHemoglobin',
                              'BMXARMC' : 'ArmCircum',
                              'BMDAVSAD' : 'SaggitalAbdominal',
                              'MGDCGSZ' : 'GripStrength',
                              'DRABF' : 'Breast_fed'})

        df_all = df_all.loc[:, ['ID', 'Gender', 'Years_in_US', 'Family_income','GlycoHemoglobin', 'ArmCircum',
                    'SaggitalAbdominal', 'GripStrength', 'Breast_fed']]

    # df_all.describe()

    # pdb.set_trace()

    observations_and_condition_df = df_all

    try:
        observations_and_condition_df.to_pickle(out_file)
    except Exception as e:
        print (e.args)
        sys.exit(1)
    else:
        print ("{} saved.".format(out_file))


# ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
def main():
    args = getting_arg().parse_args()

    if (any(vars(args).values()) == None):
        print ("Please, give me input arguments, thanks.")
        sys.exit(1)
    else:
        print (vars(args))

    if (args.choice == 1):
        syn(args.out_file)
    elif (args.choice == 2):
        input_files = list()
        for i in glob.glob(os.path.join(args.in_folder, '*.csv')):
            input_files.append(i)
        if (len(input_files) == 0) or (args.out_file == None):
            print("Please, check input parameters, thanks.")
        else:
            nhanes(input_files, args.out_file)


if __name__ == "__main__":
  main()
