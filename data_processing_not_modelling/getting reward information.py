import pandas as pd
import numpy as np
import os
from os import listdir

from datetime import datetime


def convert_date_format(date_string):
    # Define the current format
    current_format = "%Y-%m-%d_%Hh%M.%S.%f"

    # Convert to a datetime object
    datetime_object = datetime.strptime(date_string, current_format)

    # Define the desired format
    desired_format = "%Y-%m-%d %H:%M:%S.%f"

    # Convert to the desired format
    new_date_string = datetime_object.strftime(desired_format)

    return new_date_string







directory = "C:/Users/magda/PycharmProjects/RW_modelling/data/data_raw/"  # Replace with your directory.

list_trial=listdir(directory)

print(list_trial)

file_list=[]

for file_name in list_trial:
    filepath = os.path.join(directory, file_name)
    file_list.append(filepath)

print(file_list)

sona_id_list=[]
date_list=[]
reward_list=[]


for file_n in range(len(file_list)):
    """upload the result file as a pandas array"""
    csvFile = pd.read_csv(file_list[file_n])
    # print(csvFile)

    '''get the sona-id'''

    sona_id_col = csvFile['survey.sonaID']
    sona_id_col = sona_id_col.tolist()
    sona_id_try=sona_id_col[1]

    if type(sona_id_try)==str:

        sona_id=int(1111)

    else:

        sona_id = int(sona_id_col[1])

    sona_id_list.append(sona_id)


    """find the reward amount"""

    reward_col= csvFile['rew_both_condTotal']

    reward_col=reward_col.dropna()

    reward_col=reward_col.tolist()

    reward=reward_col[0]

    reward_list.append(reward)

    print(reward)


    """find the data"""

    date_col=csvFile['date']

    date=date_col[0]

    print(date)

    # Usage:
    date_string =date  # Replace with your date string.
    new_date_string = convert_date_format(date_string)
    print(new_date_string)

    date_list.append(new_date_string)


"""saving the results in csv file"""

dict = {'SONA_id': sona_id_list, 'Reward': reward_list, 'Date': date_list}

df = pd.DataFrame(dict)

# saving the dataframe
nam='C:/Users/magda/PycharmProjects/RW_modelling/data/participants_dates_rewards.csv'

df.to_csv(nam)









