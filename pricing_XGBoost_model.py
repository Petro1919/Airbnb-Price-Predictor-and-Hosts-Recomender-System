import numpy as np
import pandas as pd
import re
from nltk.tokenize import TreebankWordTokenizer

import matplotlib
import sklearn
from IPython.core.display import display, HTML

######################################################################################################################

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", font_scale=1.9)

######################################################################################################################

# TO WORK WITH
import pandas as pd
import numpy as np
from numpy import set_printoptions

# HIDE WARNINGS
import warnings
warnings.filterwarnings('ignore')

# PREPROCESSING & MODEL SELECTION
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from scipy.stats import randint
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression, LinearRegression, RidgeCV, LassoCV, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import SCORERS
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

# print(SCORERS.keys())

# PLOTTING
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("notebook")
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from sklearn import tree
from graphviz import Source
from matplotlib.pylab import rcParams
import matplotlib.lines as mlines
from yellowbrick.classifier import PrecisionRecallCurve
from yellowbrick.classifier import ROCAUC
from yellowbrick.classifier import ConfusionMatrix
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.cluster import InterclusterDistance
import plotly.express as px
import scipy.cluster.hierarchy as sch
from sklearn.metrics import classification_report


# STANDARD MODELS
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# ENSEMBLE
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

# XGBOOST
from xgboost import XGBClassifier
import xgboost as xgb

# CLUSTERING
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

# PICKLE
from pickle import dump
from pickle import load
from sklearn.metrics import mean_squared_error



# In[2]:


def split(data):
    '''
    Clean the data and return an array for the target variable and all the input variables

    '''
    data=data.dropna()

    data=data[data["price_per_night"]>=12]


    data=data[data["price_per_night"]<850]
    data=data[data["number_of_reviews"]>=1]
    data=data[data["guests_included"]>=1]



    data['extra_price']=data['security_deposit']+data['cleaning_fee']+data['extra_people']
    data['extra_price'].describe()


    for test in data['extra_price']:

        if test<25:
            data['extra_price']=25


        if 25<test and test<100:
            data['extra_price']=75


        if 100<test and test<235:
            data['extra_price']=125


        if 235<test:
            data['extra_price']=235

    scaler=MinMaxScaler(feature_range=(0,1))
    data[['extra_price']] = scaler.fit_transform(data[['extra_price']])


    for categorical_feature in ['neighbourhood_cleansed',"property_type","room_type"]:
        data = pd.concat([data,pd.get_dummies(data[categorical_feature], prefix=categorical_feature, prefix_sep='_',)], axis=1)


    data= data.drop(['neighbourhood_cleansed', 'property_type', 'room_type','description','host_about_bool',
                    'amenities',"house_rules","review_scores_cleanliness","review_scores_rating",'review_scores_accuracy',
                    'review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_value',
                     'id',"property_type", "host_response_rate", 'reviews_per_month','number_of_reviews'
                     ,'security_deposit','cleaning_fee','extra_people','extra_price','availability_365',"guests_included"
                     ,"host_is_superhost","transit_bool", "nosmok_bool", "fewdays_response_time", 'fewdays_response_time',
                     '1day_response_time', '1hour_response_time', 'fewhours_response_time', 'host_identity_verified',
                     'Internet_bool', 'super_strict_canc', 'moderate_cancellation', 'strict_cancellation', 'flexible_cancellation',
                     'require_guest_profile_picture', 'require_guest_phone_verification','Kitchen_boolean','Clothes_Dryer_bool'
                    ],axis=1)

    # Target variable (price_per_night)
    y = data["price_per_night"]
    data=data.drop(["price_per_night"],axis=1)

    #guests included to re
    data_1=data[["bathrooms","bedrooms","beds","accommodates"]]
    data=data.drop(["bathrooms","bedrooms","beds","accommodates"],axis=1)

    data=data.astype('bool')

    data=pd.concat([data, data_1], axis=1, sort=False)

    X = data

    return (X,y)



def XGB(X,y,airbnbPricing,date):
    '''
    Calculating RMSE with XGBoost Hyperparameters

    '''
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.2,
                max_depth = 7, alpha = 20, n_estimators = 70)

    xg_reg.fit(X_train,y_train)


    pricePred = xg_reg.predict(airbnbPricing)

    # If date is a special day (calculated in R), increase price
    if date in ["10/7/2020", "10/8/2020", "10/9/2020", "10/10/2020", "10/14/2020", "10/15/2020", "10/16/2020", "10/17/2020",
     "10/18/2020", "10/21/2020", "10/22/2020", "10/23/2020", "10/26/2020", "10/27/2020", "10/28/2020", "10/29/2020",
     "4/13/2020", "4/14/2020", "4/15/2020", "4/16/2020", "4/17/2020", "4/18/2020", "4/19/2020", "4/20/2020", "4/21/2020",
     "4/22/2020", "6/8/2020", "6/9/2020", "6/10/2020", "6/11/2020", "6/12/2020", "6/13/2020", "6/14/2020", "6/15/2020",
     "6/16/2020", "6/17/2020", "6/30/2020", "7/1/2020", "7/7/2020", "7/8/2020", "7/12/2020", "7/13/2020", "7/14/2020",
     "7/15/2020", "7/17/2020", "7/21/2020", "7/22/2020", "7/24/2020", "7/25/2020", "7/26/2020", "7/27/2020", "7/28/2020",
     "7/29/2020", "7/30/2020", "7/31/2020", "8/1/2020", "8/2/2020", "8/4/2020", "8/5/2020", "8/10/2020", "8/11/2020",
     "8/12/2020", "8/14/2020", "8/16/2020", "8/17/2020", "8/18/2020", "8/19/2020", "8/21/2020", "8/23/2020", "8/25/2020",
     "8/26/2020", "8/28/2020", "8/29/2020", "9/1/2020", "9/2/2020", "9/3/2020", "9/4/2020", "9/5/2020"]:
        pricePred=pricePred*1.5

    # If WEEKEND increase price by 1.2
    if date in ["1/3/2020", "1/4/2020", "1/10/2020", "1/11/2020", "1/17/2020", "1/18/2020", "1/24/2020",
    "1/25/2020", "1/31/2020", "2/1/2020", "2/7/2020", "2/8/2020", "2/14/2020", "2/15/2020", "2/21/2020",
    "2/22/2020", "2/28/2020", "2/29/2020", "3/6/2020", "3/7/2020", "3/13/2020", "3/14/2020", "3/20/2020",
    "3/21/2020", "3/27/2020", "3/28/2020", "4/3/2020", "4/4/2020", "4/10/2020", "4/11/2020", "4/17/2020",
    "4/18/2020", "4/24/2020", "4/25/2020", "5/1/2020", "5/2/2020", "5/8/2020", "5/9/2020", "5/15/2020",
    "5/16/2020", "5/22/2020", "5/23/2020", "5/29/2020", "5/30/2020", "6/5/2020", "6/6/2020", "6/12/2020",
    "6/13/2020", "6/19/2020", "6/20/2020", "6/26/2020", "6/27/2020", "7/3/2020", "7/4/2020", "7/10/2020",
    "7/11/2020", "7/17/2020", "7/18/2020", "7/24/2020", "7/25/2020", "7/31/2020", "8/1/2020", "8/7/2020",
    "8/8/2020", "8/14/2020", "8/15/2020", "8/21/2020", "8/22/2020", "8/28/2020", "8/29/2020", "9/4/2020",
    "9/5/2020", "9/11/2020", "9/12/2020", "9/18/2020", "9/19/2020", "9/25/2020", "9/26/2020", "10/2/2020",
    "10/3/2020", "10/9/2020", "10/10/2020", "10/16/2020", "10/17/2020", "10/23/2020", "10/24/2020",
    "10/30/2020", "10/31/2020", "11/6/2020", "11/7/2020", "11/13/2020", "11/14/2020", "11/20/2020",
    "11/21/2020", "11/27/2020", "11/28/2020", "12/4/2020", "12/5/2020", "12/11/2020", "12/12/2020",
    "12/18/2020", "12/19/2020", "12/25/2020", "12/26/2020"]:
        pricePred=pricePred*1.2

    return pricePred[0]



def XGB_RMSE(X,y):
    '''
    Calculating RMSE with XGBoost Hyperparameters

    '''
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.2,
                max_depth = 7, alpha = 20, n_estimators = 70)

    xg_reg.fit(X_train,y_train)

    preds = xg_reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print("RMSE: %f" % (rmse))
    return


##############################     MAIN     ##############################

data = pd.read_csv("listings_8.csv")
data.room_type.unique()
X,y= split(data)


airbnbPricing = X.iloc[1].to_frame().transpose()
airbnbPricing = airbnbPricing.astype(bool)

airbnbPricing["bathrooms"] = airbnbPricing["bathrooms"].astype(float)
airbnbPricing["bedrooms"] = airbnbPricing["bedrooms"].astype(float)
airbnbPricing["beds"] = airbnbPricing["beds"].astype(float)
airbnbPricing["accommodates"] = airbnbPricing["accommodates"].astype(float)
actual = y.iloc[1]

pred = XGB(X,y,airbnbPricing,"10/7/2020")
diff = pred-actual
print("actual", actual)
print("predicted", pred)
print("difference", diff)

XGB_RMSE(X,y)
