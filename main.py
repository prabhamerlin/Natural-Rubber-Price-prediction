
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns

price_df = pd.read_csv("rubberprice.csv")
price_df = price_df.drop(['Change','Price_rs','Change_dollar'], axis = 1)
# price_df = df[['Month','Rubber_type','Year','Price_rs']]


price_df = pd.get_dummies(price_df, columns=['Rubber_type'], prefix=['Rubber_type'])

# pd.set_option('display.max_columns', None)
# print(price_df.head())
# print(x.head())
# print(x.info())
# print(x.shape)
# print(x.describe())
# print(x.values)
# print(x.columns)
# print(x.index)




crudeoil_df = pd.read_csv("crudeoil.csv")
crudeoil_df = crudeoil_df.drop(['Date','Open','High','Low','Volume','Chg%'], axis = 1)

dollarinr_df = pd.read_csv("dollar-inr.csv")
dollarinr_df = dollarinr_df.drop(['Date','Open','High','Low','Volume','Chg%'], axis = 1)

prod_consump_import_export_df = pd.read_csv("prod_consump_import_export_new.csv")

final_df = price_df.merge(crudeoil_df, on=['Year', 'Month'])
final_df = final_df.merge(dollarinr_df, on=['Year', 'Month'])
final_df = final_df.merge(prod_consump_import_export_df, on=['Year', 'Month'])

# print(final_df.head(140))

final_df['Price_crude'] = final_df['Price_crude'].str.replace(',', '').astype('float')

# print(final_df.columns)
# print(final_df.info())


X = final_df[['Rubber_type_ISNR50', 'Rubber_type_RSS1',
       'Rubber_type_RSS4', 'Price_crude', 'Price_dollar', 'Production_tonne',
       'Consumption', 'Import_tonne', 'Export_tonne']]
y = final_df['Price_dollar_rubber']

# print(X.columns)
# print(X.info())
pd.set_option('display.max_columns', None)
print(final_df.head())

SEED = 123


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

from sklearn.model_selection import cross_val_score

MSE_CV = - cross_val_score(dt, X_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs = -1)

# dt = DecisionTreeRegressor(max_depth=6, min_samples_leaf=0.1, random_state=SEED)


dt = DecisionTreeRegressor(max_depth=None, splitter='best', min_samples_leaf=0.1, random_state=SEED)#min_samples_leaf=0.1 => each leaf has atleast 10% of training data

dt.fit(X_train, y_train)
y_predict_train = dt.predict(X_train) #predict using training set
y_predict_test = dt.predict(X_test) #predict using test set

price = dt.predict([[0, 1, 0, 3263.0, 72.534, 53000.0, 102500.0, 31368.0, 1139.0]]) #predict for a single input data
print("Price of rubber:")
print(price)

fig = plt.figure(figsize=(15, 10))
plt.style.use('seaborn')
tree.plot_tree(dt, filled=True, feature_names=X_train.columns, class_names=y_train)
fig.savefig("decision_tree1.png")
plt.show()




