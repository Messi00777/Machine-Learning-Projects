from Preprocessing import *

features = list(data.select_dtypes(include='object').columns)
print(features)

features = features[:-1]
print(features)

#Encoding Data
for i in features:
    data[i] = data[i].astype('category')
    data[i] = data[i].cat.codes
    data[i] = data[i].astype('float64')

data['y'].mask(data['y'] == 'no', 0, inplace=True)
data ['y'].mask(data['y'] == 'yes', 1, inplace=True)
data ['y'] = data['y'].astype('float64')
print(data)

sns.set_theme(style='white')
corre = data.corr()
mask = np.triu(np.ones_like(corre, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corre, mask=mask, cmap=cmap, vmax=1, vmin=1, center=0, square=True,  linewidth=.5, cbar_kws={"shrink": .5})
#plt.show()

#Looking for correlation amongst features
data_corr = data.drop('y', axis=1)
matrix = data_corr.corr()
matrix = matrix.abs().unstack()
matrix = matrix.sort_values(ascending=False)
matrix = matrix[matrix>=0]
matrix = matrix[matrix<1]
matrix = pd.DataFrame(matrix).reset_index()
matrix.columns = ['Feature_1', 'Feature_2', 'Correlation']
print(matrix.head())


 # Splitting the data
X = data.drop('y', axis=1)
Y = data['y']

# Building Decision Tree
roc_values =[]
cv = StratifiedKFold(n_splits=20)
for i in X.columns:
    roc_temp = []
    X_ = X[i].copy()
    for train, test in cv.split(X_,Y):
        clf = DecisionTreeClassifier()
        clf.fit(X_.iloc[train].fillna(0).to_frame(), Y.iloc[train])
        y_score = clf.predict_proba(X_.iloc[test].fillna(0).to_frame())
        roc_temp.append(roc_auc_score(Y.iloc[test], y_score[:,1]))
        roc_values.append(np.array(roc_temp).mean())
print(len(roc_values))

'''
roc_table = pd.DataFrame({'features':X.columns, 'mean_roc_auc_score': roc_values})
roc_table=roc_table.reset_index(drop=True)
roc_table = roc_table.sort_values(by=['mean_roc_auc_score'], ascending=False)
print(roc_table)
'''