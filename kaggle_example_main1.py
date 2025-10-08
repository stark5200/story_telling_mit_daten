import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# copied from a kaggle notebook, not sure if it works
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
# First importing the necessary libraries;
import seaborn as sns
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,RandomizedSearchCV
from sklearn.metrics import classification_report

# Adjusting screen settings;
from warnings import filterwarnings
filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')

df = pd.read_csv(r"../input/ais-dataset/ais_data.csv")
df.head()

# Looking for shape;
df.shape  

# Removing the unneeded column;
df.drop(['Unnamed: 0'], axis=1, inplace=True)

# we have lots of null values;
df.isnull().sum() 

df.info()

def grab_col_names(dataframe, cat_th=10, car_th=20):

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car
  
cat_cols, num_cols, cat_but_car = grab_col_names(df)

df.groupby('mmsi').count().shape

# the other important variable
df['navigationalstatus'].value_counts() 

def missing_values_table(data):
    m=data.isnull().sum()
    print(pd.DataFrame({'n_miss' : m[m!=0],'ratio' : m[m!=0]/len(data)}))

missing_values_table(df)

def thresholds(col, data, d, u):
    q3=data[col].quantile(u)
    q1=data[col].quantile(d)
    down=q1-(q3-q1)*1.5
    up=q1+(q3-q1)*1.5
    return down, up

def check_outliers(col, data, d=0.25, u=0.75, plot=False):
    down, up = thresholds(col, data, d, u)
    ind = data[(data[col] < down) | (data[col] > up)].index
    if plot:
        sns.boxplot(x=col, data=data)
        plt.show()
    if len(ind)!= 0:
        print(f"\n Number of outliers for '{col}' : {len(ind)}")
        return col

for col in num_cols:
    check_outliers(col, df, 0.01, 0.99) # we set thresholds 0.01 and 0.99 !!
    
def corr_analyzer(data, corr_th=0.7, plot=False):
    corr_matrix = pd.DataFrame(np.tril(data.corr(), k=-1), columns=data.corr().columns, index=data.corr().columns)
    corr = corr_matrix[corr_matrix>corr_th].stack()
    print(corr)
    if plot: # Görsel olarak analiz
        sns.heatmap(corr_matrix, cmap='Greys')
        plt.show()
    return corr[corr>corr_th].index.tolist()

corr_analyzer(df, corr_th=0.75, plot=True)

# First, the filling was made according to those in the 'heading' but not in the 'cog'. So we have fewer missings at 'cog'
df['route'] = np.where(df['cog'].isnull(), df['heading'], df['cog']) 

# Secondly, we divided the 360-degree route into 8 regions.
rot= [-1, 45, 90, 135, 180, 225, 270, 315, 360]
df['waypoint'] = pd.cut(df['route'], rot, labels=['NNE','ENE','ESE','SSE','SSW','WSW','WNW','NNW'])

# Finally, the ships with less than 5.5kts speed and no route information were tagged as 'FIX'.
df['waypoint'] = np.where((df['sog']<5.5) & (df['waypoint'].isnull()), 'FIX', df['waypoint'])

df['speed'] = df["sog"].fillna(df.groupby(['shiptype', 'waypoint'])['sog'].transform('mean'))

df['dimension'] = df['width'] * df['length']

df = df.drop_duplicates(subset='mmsi') # one vessel has more than one data
df.groupby(['shiptype'])['mmsi'].count()

# no need anymore
df.drop(['cog', 'heading', 'route', 'sog', 'mmsi'], axis=1, inplace=True) 

# taking account for the new variables
cat_cols, num_cols, cat_but_car = grab_col_names(df) 

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()
    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]
    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])
    return temp_df

df = rare_encoder(df, 0.02) # Less than %2 ratio will be grouped

# combining the 'Unknown value' and 'Rare' groups at 'navigationalstatus' variable
df['status'] = df['navigationalstatus'].where(
    ~((df['navigationalstatus']=='Unknown value') | (df['navigationalstatus']=='Rare')), 'Other') 

df.drop(['navigationalstatus', 'width', 'length'], axis=1, inplace=True)
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# with rare class
for col in cat_cols:
    cat_summary(df, col)
    
# Splitting data as an output and predictors;
y = df['shiptype']
X= df.drop('shiptype', axis=1)

# One hot encoder process;
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

X = one_hot_encoder(X, ['waypoint', 'status'])
X.head(3)

scaler = RobustScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# missing values left. But we will use Light GBM, so it will not affect our model  
missing_values_table(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

lgbm_model = LGBMClassifier(random_state=17) # basic model
lgbm_model

lgbm_param={'max_depth' : [2, 5, 8, 10],
            'learning_rate' : [0.001, 0.01, 0.05, 0.1],
            'n_estimators' : [200, 400, 600, 900],
            'colsample_bytree' : [0.3, 0.5, 0.7, 1]}

lgbm_cv = GridSearchCV(lgbm_model, lgbm_param, cv=10, n_jobs=-1, verbose=True)
lgbm_cv.fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_cv.best_params_, random_state=17).fit(X_train, y_train)


lgbm_results = cross_val_score(lgbm_final, X_test, y_test, cv=10, scoring="accuracy").mean()
lgbm_results

y_pred = lgbm_final.predict(X_test)
print(classification_report(y_test, y_pred))

def plot_importance(model, features, num=len(X)):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    
plot_importance(lgbm_final, X_train)

random_ship = X.sample(1)
random_ship# this is our one sample to predict

def estimator(sample):
   print(f"Our Actual Ship Type \t: {lgbm_final.predict(random_ship)[0]}")
   temp = y.reset_index()
   print(f"Our Predicted Ship Type \t: {temp[temp.index==sample.index[0]].iloc[0,1]}")

estimator(random_ship)