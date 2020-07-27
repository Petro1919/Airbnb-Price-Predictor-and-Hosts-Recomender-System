# IMPORTING EVERYTHING I COULD POSSIBLY NEED #

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import re
from nltk.tokenize import TreebankWordTokenizer

import matplotlib
import sklearn
from IPython.core.display import display, HTML

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", font_scale=1.9)

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


def recommendations(coefs, airbnb, X_wnei, data, recom):
    '''
    Makes recommendations based on the specific Airbnb's characteristics and the important features recognized by
    the best model in best_models()

    '''

    X_wnei_cols = X_wnei.columns

    for_improvement = []
    for col in range(1,len(X_wnei.columns)):
        if coefs[col] > 0 :
            if airbnb[0][col] < data.iloc[:,col].mean():
                if "cancellations" not in recom.iloc[0, col] and ("You should be more moderate on cancellations" not in for_improvement and "You should be less strict on cancellations" not in for_improvement):
                    if "messages" not in recom.iloc[0, col] and ("You should answer messages sooner" not in for_improvement):
                        sent = recom.iloc[0, col]
                        for_improvement.append(sent)
        elif coefs[col] < 0:
            if airbnb[0][col] > data.iloc[:,col].mean():
                if "cancellations" not in recom.iloc[1, col] and ("You should be more moderate on cancellations" not in for_improvement and "You should be less strict on cancellations" not in for_improvement):
                    if "messages" not in recom.iloc[1, col] and ("You should answer messages sooner" not in for_improvement):
                        sent = recom.iloc[1, col]
                        for_improvement.append(sent)

    # If there are improvements to make...
    if len(for_improvement) != 0:
    #   Concatenating messages
        mess1 = "In order for your Airbnb to be truly competitive, make sure you do the following things: "
        mess2 = "\n".join(for_improvement)
        message = mess1 + "\n" + mess2

        return print(message)
    # If there are NO improvements to make...
    else:
        return print("Your Airbnb is very competitive compared to all the other Airbnbs in Boston!")




def best_models(data):
    '''
    Implements Ridge, Lasso and ElasticNet to determine best model, based on inputted data.

    '''
    MSEs = []

    data=data.dropna()

    data=data[data["number_of_reviews"]>=1]
    data=data[data["guests_included"]>=1]

    data['price_per_person'] = data['price_per_night']/(data['guests_included']+data['extra_people'])

    data['output'] = (data['number_of_reviews']*data['review_scores_rating'])/data['availability_365']

    data = data.drop(["property_type","room_type","description","house_rules", "amenities", "id",
    "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
    "review_scores_checkin", "review_scores_communication", "review_scores_value", "number_of_reviews",
                      "reviews_per_month", "availability_365", "neighbourhood_cleansed","bathrooms",
                      "host_is_superhost", "accommodates", "bedrooms","beds", "guests_included","price_per_night",
                      "extra_people", "host_is_superhost", "Kitchen_boolean", "Gym_bool", "Elevator_in_building_bool",
                      "Clothes_Washer_bool", "Internet_bool", "price_per_person"],axis=1)

    scaler=MinMaxScaler(feature_range=(0,1))


    data[['security_deposit']] = scaler.fit_transform(data[['security_deposit']])
    data[['cleaning_fee']] = scaler.fit_transform(data[['cleaning_fee']])
    data[['host_response_rate']] = scaler.fit_transform(data[['host_response_rate']])

    data.to_numpy()
    data=pd.DataFrame(data)

    data=data.dropna()

    data['output'][data['output'] > 260] = 260

    y = data["output"] # Target variable (price)
    X_wnei = data.drop(["output"],axis=1)

    # Creating new DF without neighborhood names
    X_wnei.to_csv("X_wnei.csv", index=False)
    data.to_csv("data.csv", index=False)

    #########################################################################################################
    # Ridge Regression

    kfold=KFold(n_splits=10, random_state=7)

    model=Ridge()
    scoring = "neg_mean_squared_error"

    results=cross_val_score(model, X_wnei, y, cv=kfold, scoring=scoring)
    clf = model.fit(X_wnei, y)
    MSEs.append(("Ridge Regression", results.mean(), clf.coef_))


    #########################################################################################################
    # Lasso Regression

    kfold=KFold(n_splits=10, random_state=7)

    model=Lasso()
    scoring = "neg_mean_squared_error"

    results=cross_val_score(model, X_wnei, y, cv=kfold, scoring=scoring)
    clf = model.fit(X_wnei, y)
    MSEs.append(("Lasso Regression", results.mean(), clf.coef_))

    #########################################################################################################
    # Elastic Net Regression

    kfold=KFold(n_splits=10, random_state=7)

    model=ElasticNet()
    scoring = "neg_mean_squared_error"

    results=cross_val_score(model, X_wnei, y, cv=kfold, scoring=scoring)
    clf = model.fit(X_wnei, y)
    MSEs.append(("ElasticNet Regression", results.mean(), clf.coef_))

    #########################################################################################################

    return (min(MSEs, key = lambda t: t[1])[0], X_wnei, y, min(MSEs, key = lambda t: t[1])[2], data)


######################    MAIN    ######################


data = pd.read_csv("listings_8.csv")
recom = pd.read_csv("recom.csv")

best_model = best_models(data)

method = best_model[0]
X_wnei = best_model[1]
y = best_model[2]
coefs = best_model[3].reshape(33,1)
data = best_model[4]


# Using an airbnb from our data so that we don't have to create one manually
# If you want to change the airbnb, chnage the iloc index to another one (in our data range)

# Example of airbnb that doesn't need any improvements according to our best model
print("\nExample of airbnb that doesn't need improvements according to our best model")
print("-")
airbnb = pd.read_csv("X_wnei.csv").iloc[1345].values.reshape(1,-1)
recommendations(coefs, airbnb, X_wnei, data, recom)
print("____________________________________________________________________________________\n")

# Example of airbnb that needs improvements according to our best model
print("\nExample of airbnb that needs improvements according to our best model")
print("-")
airbnb = pd.read_csv("X_wnei.csv").iloc[1].values.reshape(1,-1)
recommendations(coefs, airbnb, X_wnei, data, recom)
print("____________________________________________________________________________________\n")



# Good examples to showcase!
#
# airbnb = pd.read_csv("X_wnei.csv").iloc[1345].values.reshape(1,-1) -> Nothing to improve
# airbnb = pd.read_csv("X_wnei.csv").iloc[1].values.reshape(1,-1) -> A lot to improve
