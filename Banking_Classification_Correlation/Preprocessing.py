from Libraries_Import import *


print(data.shape)
print(predict.shape)
print(data.columns)

data = data.drop(["previous", "pdays"], axis=1)
print(data.head())
print(data.info())

data_num = data.drop(['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome', 'y'], axis=1)
print(data_num.describe().T)

# Jobs
plt.figure(figsize=(12,6))
a = sns.countplot(y='job', data =data, order=data['job'].value_counts().index)
a.axes.set_title('Jobs', fontsize= 22)
a.set_ylabel('Count', fontsize=16)
a.tick_params(labelsize=12)
#plt.show()

# Marital Stats
plt.figure(figsize=(12,6))
a = sns.countplot(x='marital', data=data, order=data['marital'].value_counts().index)
a.axes.set_title('Marital Stats', fontsize=22)
a.set_ylabel('Count', fontsize=15)
a.tick_params(labelsize=12)
#plt.show()

#Education Stats
plt.figure(figsize=(12,6))
a = sns.countplot(y='education', data=data, order=data['education'].value_counts().index)
a.axes.set_title('Education Stats', fontsize=22)
a.set_ylabel('Count', fontsize=15)
a.tick_params(labelsize=12)
#plt.show()

#Housing Stats
plt.figure(figsize=(12,6))
a = sns.countplot(x='housing', data=data, order=data['housing'].value_counts().index)
a.axes.set_title('Housing Stats', fontsize=22)
a.set_ylabel('Count', fontsize=15)
a.tick_params(labelsize=12)
#plt.show()

# Previous Campaign Outcomes
plt.figure(figsize=(12,6))
a = sns.countplot(x='poutcome', data=data, order=data['poutcome'].value_counts().index)
a.axes.set_title('Previous Campaign Outcome', fontsize=22)
a.set_ylabel('Count', fontsize=15)
a.tick_params(labelsize=12)
#plt.show()

# Subscribers
plt.figure(figsize=(12,6))
a = sns.countplot(x='y', data=data, order=data['y'].value_counts().index)
a.axes.set_title('Subscribers', fontsize=22)
a.set_ylabel('Count', fontsize=15)
a.tick_params(labelsize=12)
#plt.show()

# Checking null values
#print(data.isnull().count)
print(data.isnull().sum())

# Dropping Duplicated Values
data = data.drop_duplicates()
print(len(data))

