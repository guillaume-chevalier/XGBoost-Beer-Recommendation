
# Beer recommendation

## Where to place new data

New data should be placed under the subfolder "./new-beer-recipes-here/*". 

IMPORTANT: The old data should still be placed under the subfolder "./original-beer-recipes/*" here, because the program needs to re-lear at each startup (I don't persist the learning to save time). Training may take some time (less than 5 minutes).

## How to use

Run this notebook with "ipython notebook", or "jupyter notebook", python 3 version. 

## Note on corrupted data

The line 4070 (ID 4069) of the CSV is corrupted and doesn't have proper alignment of columns for some reason: 
`4069,DIPA, mango smoothie,/homebrew/recipe/view/435411/dipa-mango-smoothie,Double IPA,56,18.93,1.084,1.011,9.52,64.48,5.24,15.14,30,1.104,75,N/A,Specific Gravity,BIAB,0.5,21.11,Snowberry Honey,N/A,10,`. For example, when I ask for the column "StyleID" that should give me the value "56" here, I instead get the previous column "Double IPA". It is because there is 3 fields instead of 2 at the beginning, the error is here: `DIPA, mango smoothie` should not contain any comma. I fixed the line by deleting this comma instead of deleting the line: to keep more data for training. 

Note: There is also an extra comma at the end of every line, which creates an empty column. I ignore this column when processing the data. So the input is all first columns, output is the last column, and the next last column is ignored and deleted. 

## We'll now load the data set in Python and then train decision trees with XGBoost


```python
# python3
# pip install xgboost
# pip install numpy
# pip install matplotlib
# pip install scikit-learn
# pip install pandas

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_tree
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import GridSearchCV

import os
import glob


# xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
#                            colsample_bytree=1, max_depth=7)

```


```python
def load_categories():
    relative_data_path = './original-beer-recipes/styleData.csv'
    pdtrain = pd.read_csv(relative_data_path, index_col='BeerID', encoding='latin1')
    return pdtrain

def remove_last_columns(df):
    while not list(df.columns)[-1] in ["UserId", "rating"]:
        df.drop(df.columns[len(df.columns)-1], axis=1, inplace=True)
    return df

def cast_column_to(df, colname, _type):
    
    # df.transform(lambda x: x.fillna(x.mean()))

    df[colname] = df[colname].astype(_type)
    
    return df

def load_data_to_pd(relative_data_path='./original-beer-recipes/recipeData.csv'):
    
    target = None  # might there be no target at test time?
    
    pdtrain = pd.read_csv(relative_data_path, index_col='BeerID', encoding='latin1')
    # print(pdtrain.columns)
    # Index(['Name', 'URL', 'Style', 'StyleID', 'Size(L)', 'OG', 'FG', 'ABV', 'IBU',
    #    'Color', 'BoilSize', 'BoilTime', 'BoilGravity', 'Efficiency',
    #    'MashThickness', 'SugarScale', 'BrewMethod', 'PitchRate', 'PrimaryTemp',
    #    'PrimingMethod', 'PrimingAmount', 'UserId', 'rating', '\t'],
    #   dtype='object')
    pdtrain = remove_last_columns(pdtrain)
    
    pdtrain = pdtrain.drop(['URL'], axis=1)
    pdtrain = pdtrain.drop(['Name'], axis=1)
    
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        pass #print(pdtrain['StyleID'])
        # pdtrain['StyleID'] = pdtrain['StyleID'].astype(int)
    pdtrain = pdtrain.drop(['StyleID'], axis=1)
    pdtrain = pdtrain.drop(['Style'], axis=1)
    
    pdtrain = cast_column_to(pdtrain, 'Size(L)', float)
    pdtrain = cast_column_to(pdtrain, 'OG', float)
    pdtrain = cast_column_to(pdtrain, 'FG', float)
    pdtrain = cast_column_to(pdtrain, 'ABV', float)
    pdtrain = cast_column_to(pdtrain, 'IBU', float)
    pdtrain = cast_column_to(pdtrain, 'Color', float)
    pdtrain = cast_column_to(pdtrain, 'BoilSize', float)
    pdtrain = cast_column_to(pdtrain, 'BoilGravity', float)
    pdtrain = cast_column_to(pdtrain, 'Efficiency', float)
    pdtrain = cast_column_to(pdtrain, 'Color', float)
    pdtrain = cast_column_to(pdtrain, 'MashThickness', float)
    
    # todo: SugarScale, BrewMethod PrimingMethod'PrimingAmount'
    pdtrain = pdtrain.drop(['SugarScale'], axis=1)
    pdtrain = pdtrain.drop(['BrewMethod'], axis=1)
    # pdtrain = cast_column_to(pdtrain, 'PitchRate', float)
    pdtrain = pdtrain.drop(['PitchRate'], axis=1)
    # pdtrain = cast_column_to(pdtrain, 'PrimaryTemp', float)
    pdtrain = pdtrain.drop(['PrimaryTemp'], axis=1)
    pdtrain = pdtrain.drop(['PrimingMethod'], axis=1)
    pdtrain = pdtrain.drop(['PrimingAmount'], axis=1)
    # pdtrain = cast_column_to(pdtrain, 'UserId', int)
    pdtrain = pdtrain.drop(['UserId'], axis=1)
    pdtrain = cast_column_to(pdtrain, 'rating', float)
    
    # bad cols: 
    # StyleID, SugarScale, BrewMethod, PitchRate, PrimingMethod, PrimingAmount, UserId
    
    # Style    ,StyleID, Size(L),   OG,FG   ,ABV ,IBU  ,Color,BoilSize,BoilTime,BoilGravity,Efficiency,MashThickness,
    # Cream Ale,45     , 21.77  ,1.055,1.013,5.48,17.65,4.83 ,28.39   ,75      ,1.038      ,70        ,N/A          ,
    
    # SugarScale      ,BrewMethod,PitchRate,PrimaryTemp,PrimingMethod,PrimingAmount,UserId,rating,
    # Specific Gravity,All Grain ,N/A      ,17.78      ,corn sugar   ,4.5 oz       ,116   ,6,
    # Specific Gravity,All Grain, N/A,      N/A        ,corn sugar   ,4.2 oz,        116,0,
    # Specific Gravity,BIAB     ,0.35       ,17        ,sucrose        ,140 g,82450,0,


    try: 
        pdtrain = pdtrain[np.isfinite(pdtrain['rating'])]
        # pd.DataFrame.dropna(pdtrain).shape
        target = pd.DataFrame(pdtrain['rating'])
        pdtrain = pdtrain.drop(['rating'], axis=1)
    except: 
        print("No test data found in last column named 'rating': can't evaluate!!!")
        pass
    
    # pdtrain.head()
    
    # print(pdtrain.columns)
    # print(target.columns)
    
    # print(pdtrain.rating)
    # print(pdtrain["Unnamed: 24"])

    # print("Data set's shape,")
    # print("X.shape, integer_y.shape, len(attrs_names), len(output_labels):")
    # print(X.shape, integer_y.shape, len(attrs_names), len(output_labels))
    # (1728, 27) (1728,) 27 4

    # # Shaping the data into a single pandas dataframe for naming columns:
    # pdtrain = pd.DataFrame(X)
    # pdtrain.columns = attrs_names
    # dtrain = xgb.DMatrix(pdtrain, integer_y)
    
    # xgtrain = xgb.DMatrix(train.values, target.values)
    dtrain = xgb.DMatrix(pdtrain, target)
    
    target = np.array(target.values).flatten()
    return pdtrain, target, dtrain

pdtrain, target, dtrain = load_data_to_pd()
print(pdtrain.values.shape)

```

    (73691, 11)


    /usr/local/lib/python3.6/dist-packages/IPython/core/interactiveshell.py:2903: DtypeWarning: Columns (4,18,22) have mixed types. Specify dtype option on import or set low_memory=False.
      if self.run_code(code, result):


### Define the features and preprocess the car evaluation data set

We'll preprocess the attributes into redundant features, such as using an integer index (linear) to represent a value for an attribute, as well as also using a one-hot encoding for each attribute's possible values as new features. Despite the fact that this is redundant, this will help to make the tree smaller since it has more choice on how to split data on each branch. 

### Train simple decision trees (here using XGBoost) to fit the data set:

First, let's define some hyperparameters, such as the depth of the tree.


```python
"""
xgb1 = xgb.XGBRegressor()

parameters = {
    "max_depth": [1, 3],
    "n_estimators": [1, 3],
    "learning_rate": [0.9, 1]
    # "reg_alpha": [0, 1e-2, 1],
    # "reg_lambda": [0, 1e-2, 1]
}

import sklearn
xgb_grid = GridSearchCV(xgb1,
                        parameters,
                        cv = 2,
                        verbose=True, 
                       scoring=sklearn.metrics.make_scorer(sklearn.metrics.r2_score))

# pdtrain = pdtrain.fillna(pdtrain.median()).clip(-1e6,1e6)
# pdtrain = pdtrain.reset_index()

pdtrain = pdtrain.fillna(0)
xgb_grid.fit(pdtrain.values, target)
"""
```




    '\nxgb1 = xgb.XGBRegressor()\n\nparameters = {\n    "max_depth": [1, 3],\n    "n_estimators": [1, 3],\n    "learning_rate": [0.9, 1]\n    # "reg_alpha": [0, 1e-2, 1],\n    # "reg_lambda": [0, 1e-2, 1]\n}\n\nimport sklearn\nxgb_grid = GridSearchCV(xgb1,\n                        parameters,\n                        cv = 2,\n                        verbose=True, \n                       scoring=sklearn.metrics.make_scorer(sklearn.metrics.r2_score))\n\n# pdtrain = pdtrain.fillna(pdtrain.median()).clip(-1e6,1e6)\n# pdtrain = pdtrain.reset_index()\n\npdtrain = pdtrain.fillna(0)\nxgb_grid.fit(pdtrain.values, target)\n'




```python
# best_parameters = xgb_grid.best_estimator_.get_params()
# print(best_parameters)
```


```python
# dir(xgb.XGBRegressor)

trees = xgb.XGBRegressor(max_depth=20, num_boost_round=10, learning_rate=1.43)
# trees = xgb.XGBRegressor(**best_parameters)

trees.fit(pdtrain, target)

got_targets = trees.predict(pdtrain)
```


```python
print("mse score (mean squared error score):")
np.array((got_targets - target)**2).mean()
```

    mse score (mean squared error score):





    0.05049972042299052



## Test on other data

for now, the other data is the same as train ! but eventually, it'll be your test data. 


```python

test_datasets_paths = glob.glob("./new-beer-recipes-here/*")

print("Here are your test data files:", test_datasets_paths)

```

    Here are your test data files: ['./new-beer-recipes-here/exampleRecipeData3.csv', './new-beer-recipes-here/exampleRecipeData2.csv', './new-beer-recipes-here/exampleRecipeData1.csv']



```python
for file in test_datasets_paths: 
    
    print("TESTING FILE:", file)
    
    # Note: loading might fail because of errors related to offset columns in CSV: during data loading, I needed to discard columns do to them missing a comma or having too many commas. weird. 
    pdtest, target_t, dtest = load_data_to_pd(file)
    
    got_targets = trees.predict(pdtest)
    
    print("mse score (mean squared error score):", np.array((got_targets - target_t)**2).mean())
    
    print("")
```

    TESTING FILE: ./new-beer-recipes-here/exampleRecipeData3.csv


    /usr/local/lib/python3.6/dist-packages/IPython/core/interactiveshell.py:2903: DtypeWarning: Columns (4,18,22) have mixed types. Specify dtype option on import or set low_memory=False.
      if self.run_code(code, result):


    mse score (mean squared error score): 0.13548294245926867
    
    TESTING FILE: ./new-beer-recipes-here/exampleRecipeData2.csv
    mse score (mean squared error score): 0.13548294245926867
    
    TESTING FILE: ./new-beer-recipes-here/exampleRecipeData1.csv
    mse score (mean squared error score): 0.13548294245926867
    

