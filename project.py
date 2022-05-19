# %%
# imports
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import tree

# %%
# Read in data
data = pd.read_csv('nasa.csv')
data["Hazardous"].value_counts()


# %%
# Plot correlations between columns
# pip install seaborn for fancy graph
plt.figure(figsize=(21, 14))
sns.heatmap(data.corr(), annot=True)
# fancy graph shows that many columns are just different
# units for same values so will need to drop those
drop_columns = ['Neo Reference ID', 'Name', 'Est Dia in KM(max)', 'Est Dia in M(min)', 'Est Dia in M(max)', 'Est Dia in Miles(min)', 'Est Dia in Miles(max)', 'Est Dia in Feet(min)', 'Est Dia in Feet(max)',
                'Relative Velocity km per hr', 'Miles per hour', 'Miss Dist.(lunar)', 'Close Approach Date', 'Epoch Date Close Approach', 'Orbit Determination Date',
                'Miss Dist.(kilometers)', 'Miss Dist.(miles)', 'Orbiting Body', 'Equinox']

# %%
# Hazardous column is True/False so turn into 1/0
cat_encoder = LabelEncoder()
label = cat_encoder.fit_transform(data[["Hazardous"]])

# %%
# todo look into type conversion more instead of dropping dates
# dates = ['Close Approach Date', 'Epoch Date Close Approach', 'Orbit Determination Date']
# for d in dates:
#     data[d] = data[d].astype(float)

# %%
# Drop data that's not needed
drop_columns.append("Hazardous")
data = data.drop(drop_columns, axis=1)
data.head()

# %%
# Plot new correlations between columns
plt.figure(figsize=(21, 14))
sns.heatmap(data.corr(), annot=True)

# %%
# Prepare data for algorithms
X_train, X_test, y_train, y_test = train_test_split(
    data, label, test_size=0.2, random_state=42)

# %%
# KNN fitting
knn = GridSearchCV( KNeighborsClassifier(),{'n_neighbors' : list(range(1, 20))},cv = 5)
knn.fit(X_train, y_train)

# %%
# KNN performance
y_pred = knn.best_estimator_.predict(X_test)

print(knn.best_estimator_)
print(metrics.classification_report(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))

y_pred_prob = knn.best_estimator_.predict_proba(X_test)[::, 1]
knnfpr, knntpr, knnthreshold = metrics.roc_curve(y_test, y_pred_prob)

# %%
# GNB fitting
gnb = GridSearchCV( GaussianNB(),{},cv = 5)
gnb.fit(X_train, y_train)

# %%
# GNB performance
y_pred2 = gnb.best_estimator_.predict(X_test)
#print(gnb.best_estimator_)
y_pred_prob2 = gnb.best_estimator_.predict_proba(X_test)[::, 1]
print(metrics.classification_report(y_test, y_pred2))
print(metrics.confusion_matrix(y_test, y_pred2))
gnbfpr, gnbtpr, gnbthreshold = metrics.roc_curve(y_test, y_pred_prob2)

# %%
# Decision Tree fitting
params = {'max_leaf_nodes': list([5, 10, 15, 20, 25]),
          'min_samples_split': list([5, 10, 15, 20, 25]),
          'max_depth': list([5, 10, 15, 20, 25])}
gscv = GridSearchCV(DecisionTreeClassifier(), params, cv=5)
gscv.fit(X_train, y_train)
print(gscv.best_estimator_)

# %%
# Decision Tree performance
y_pred = gscv.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# %%
# Plot fitted tree
plt.figure(figsize=(21, 14))
tree.plot_tree(gscv.best_estimator_, filled=True,
               feature_names=data.columns, fontsize=12)
