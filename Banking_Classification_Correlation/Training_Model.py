from matplotlib.pyplot import grid
from numpy.lib.function_base import average
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

#ROC
y_predict = grid_search.best_estimator_.predict_proba(X_test)[:,1]

#Visualizing using plot metric
Binary_class = BinaryClassification(Y_test, y_predict, labels=[1,0])
plt.figure(figsize=(7,6))
Binary_class.plot_roc_curve()
plt.title('ROC CURVE')
#plt.show()

#Recall
rec = make_scorer(recall_score, average='macro')
grid_search_recall = GridSearchCV(estimator=dt, param_grid=params, scoring= rec, cv=10, n_jobs=-1)
grid_search_recall = grid_search.fit(X_train, Y_train)
y_predict = grid_search_recall.best_estimator_.predict(X_test)

# Model Evaluation
print('best parameters: ', grid_search_recall.best_params_)
print(confusion_matrix(Y_test, y_predict))
print(classification_report(Y_test, y_predict))

# Precision
prec = make_scorer(precision_score, average='macro')
grid_search_prec= GridSearchCV(estimator=dt, param_grid=params, scoring=prec, cv=10, n_jobs=-1)
grid_search_prec = grid_search_prec.fit(X_train, Y_train)
y_predict = grid_search_prec.best_estimator_.predict(X_test)

# Model Evaluation
print('best parameters: ', grid_search_prec.best_params_)
print(confusion_matrix(Y_test, y_predict))
print(classification_report(Y_test, y_predict))

def plot_grid_search(cv_results, grid_params1, grid_params2, name_params1, name_params2, title):
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_params2).reshape(len(grid_params2), len(grid_params1)))
    _,ax = plt.subplots(1,1)

    for id, val in enumerate(grid_params2):
        ax.plot(grid_params1[:-1], scores_mean[id,:-1], '-o', label= name_params2+ ':'+str(val))
        ax.plot(19, scores_mean[id, -1:], '*', label='crt'+'='+str(val)+'& mx_dpt= None')
        ax.set_title(title, fontsize=18, fontweight='bold')
        ax.set_ylabel(title+ '[CV Avg Score]', fontsize=14)
        ax.legend(loc='best', fontsize=15)
        ax.grid('on')


##  Calling Method
plot_grid_search(grid_search.cv_results_, params['max_depth'], params['criterion'], 'max_depth', 'criterion', ' Accuracy')
plot_grid_search(grid_search_recall.cv_results_, params['max_depth'], params['criterion'], 'max_depth', 'criterion', 'Recall')
plot_grid_search(grid_search_prec.cv_results_, params['max_depth'], params['criterion'], 'max_depth', 'criterion', 'Precision')
plt.show()