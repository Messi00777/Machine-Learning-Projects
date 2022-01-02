from Libraries_Import import *


print(data.shape)
print(predict.shape)
print(data.columns)

data = data.drop(["previous", "pdays"], axis=1)
print(data.head())
print(data.info())

data_num = data.drop(['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome', 'y'], axis=1)
print(data_num.describe().T)

plt.figure(figsize=(12,6))
a = sns.countplot