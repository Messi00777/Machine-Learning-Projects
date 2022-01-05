from matplotlib.pyplot import grid
from Correlation import *

#Decision Tree (Supervised Learning)

data1 = data_df.copy()
X = data1.drop('y', axis=1)
Y = data1['y']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25, random_state=45)
dt = DecisionTreeClassifier()
params = {'criterion':['gini', 'entropy'], 'max_depth':[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,None]}

# Grid Search
grid_search = GridSearchCV(estimator=dt, param_grid=params, scoring='accuracy', cv=10, n_jobs=-1)
grid_search= grid_search.fit(X_train, Y_train)
y_predict = grid_search.best_estimator_.predict(X_test)

# Model Evaluation
print('best parameters:', grid_search.best_params_)
print(confusion_matrix(Y_test, y_predict))
print(classification_report(Y_test, y_predict))


