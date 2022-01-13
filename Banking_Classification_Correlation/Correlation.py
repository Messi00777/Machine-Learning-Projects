from Feature_Selection import *
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
from pylab import rcParams
rcParams['figure.figsize']= 10,5
import warnings
warnings.filterwarnings('ignore')

# Perorming data augmentation by creating synthetic data points based on the original data using SMOTE.

print(data_original)

#encoding y
data_original['y'].mask(data_original['y'] == 'yes',1, inplace=True)
data_original['y'].mask(data_original['y'] == 'no', 0, inplace=True)

# Splitting the data to X and Y
X = data_original.drop('y', axis=1)
Y=data_original['y']
Y = Y.astype('int')

X = pd.get_dummies(X, drop_first=True)

# Balancing the data using SMOTE
smt = SMOTE()
X_smt, Y_smt, = smt.fit_resample(X,Y)

 # Convertimg the data points to dataframe
data_df = X_smt
data_df['y'] = Y_smt

print(data_df)


# checking client subscribers (checking if the data is balanced or not)
plt.figure(figsize=(5, 5))
b = sns.countplot(x ='y', data = data_df, order=data_df["y"].value_counts().index)
b.axes.set_title("has the client subscribed a term deposit?",fontsize=22)
b.set_ylabel("Count",fontsize=15)
b.tick_params(labelsize=12)
#plt.show()



