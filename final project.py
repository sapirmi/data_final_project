import csv
import os.path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# read the data + preview of the data and info of the data types in it
my_path = os.path.abspath(os.path.dirname(__file__)) + "\heart_failure_clinical_records_dataset.csv"
data = pd.read_csv(my_path)
print(data.info())
print(data.head())

# changing data types where needed
data["anaemia"] = data.anaemia.astype("category")
data["diabetes"] = data.diabetes.astype("category")
data["high_blood_pressure"] = data.high_blood_pressure.astype("category")
data["sex"] = data.sex.astype("category")
data["smoking"] = data.smoking.astype("category")
data["DEATH_EVENT"] = data.DEATH_EVENT.astype("category")

# presenting the number of death cases in each age group compare to the number of recovery cases in the same age group
# note - in DEATH_EVENT category 0=NO, 1=YES
# organizing number of cases by age group and type
live = {}
dead = {}
def num_of_cases(name, x, y):
    age_group = data.loc[(data["age"] > x) & (data["age"] < y)]
    dead_from_age_group = (age_group.loc[age_group["DEATH_EVENT"] == 1]).shape[0]
    live_from_age_group = (age_group.loc[age_group["DEATH_EVENT"] == 0]).shape[0]
    live[name] = live_from_age_group
    dead[name] = dead_from_age_group
num_of_cases("40-49", 39, 50)
num_of_cases("50-59", 49, 60)
num_of_cases("60-69", 59, 70)
num_of_cases("70-79", 69, 80)
num_of_cases("80-89", 79, 90)
num_of_cases("90+", 81, 100)

# making 1 graph that has 2 different bars, including design like size, color, spacing, presenting numbers
plt.figure(figsize=(7, 6))
width = 0.3
bar1 = np.arange(len(live.keys()))
bar2 = [i + width for i in bar1]
plt.bar(bar1, live.values(), width, label="Number of Recovered Patients", color="teal")
plt.bar(bar2, dead.values(), width, label="Number of Deceased Patients", color="navy")
plt.xticks(bar1, live.keys())
plt.title("Number of Deaths or Recovery from Heart Event According to Age Groups")
plt.xlabel("Age by Groups")
plt.ylabel("Number of Cases")
for i, v in enumerate(live.values()):
    plt.text(i - 0.1, v + 1, str(v), fontweight="bold")
for i, v in enumerate(dead.values()):
    plt.text(i + 0.2, v + 1, str(v), fontweight="bold")
plt.legend()
plt.show()

# making pie graph for each age group separately
def pie_age_group(group_name, x, y):
    age_group = data.loc[(data["age"] > x) & (data["age"] < y)]
    dead_group = (age_group.loc[age_group["DEATH_EVENT"] == 1]).shape[0]
    dead_prc = dead_group / age_group.shape[0] * 100
    dead_prc = round(dead_prc, 2)
    live_prc = round(100 - dead_prc, 2)
    total_prc = [live_prc, dead_prc]
    plt.pie(total_prc, labels = ["Recovery Cases", "Death Cases"],
            autopct = "%1.2f%%", colors = ["dodgerblue" ,"hotpink"])
    plt.title("Age Group - {}".format(group_name))

# making subplot for all of the age groups' charts + design ant titles
f, axs =plt.subplots(3, 2, figsize=(15,15))
f.suptitle("Recovery VS Death Cases by Percentage", size = 20)
plt.subplot(3, 2, 1)
pie_age_group("40-49", 39, 50)
plt.subplot(3, 2, 2)
pie_age_group("50-59", 49, 60)
plt.subplot(3, 2, 3)
pie_age_group("60-69", 59, 70)
plt.subplot(3, 2, 4)
pie_age_group("70-79", 69, 80)
plt.subplot(3, 2, 5)
pie_age_group("80-89", 79, 90)
plt.subplot(3, 2, 6)
pie_age_group("90+", 81, 100)
plt.show()
