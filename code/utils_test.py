import os
import glob
import pandas as pd
import numpy as np
import requests
import urllib
import unicodedata
from statistics import mean,pstdev
from datetime import datetime,timedelta
#import jpholiday
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose,STL
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf,pacf,adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_process import ArmaProcess
from itertools import product
from typing import Union
import japanize_matplotlib
import seaborn
#import sweetviz as sv
import lightgbm as lgb
import catboost 
from catboost import cv,CatBoostClassifier,CatBoostRegressor, Pool
import xgboost as xgb
#import category_encoders
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold,StratifiedGroupKFold,TimeSeriesSplit,cross_val_score,GroupKFold
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.preprocessing import PolynomialFeatures,LabelEncoder
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.inspection import PartialDependenceDisplay

from collections import Counter
from imblearn.over_sampling import SMOTE,SMOTENC

from scipy.optimize import minimize
#from prophet import Prophet
#from prophet.diagnostics import performance_metrics,cross_validation
#from prophet.plot import add_changepoints_to_plot
#from neuralprophet import NeuralProphet
import optuna
#from optuna.integration import OptunaSearchCV
import webbrowser
from geopy.geocoders import Nominatim,Photon
from geopy.extra.rate_limiter import RateLimiter
from geopy.distance import geodesic
from geolib import geohash
#from normalize_japanese_addresses import normalize
import folium
import geopandas
import tqdm
from tqdm import tqdm_notebook

import requests
import urllib.parse
import re
import json
import joblib
import gc
#import shap

#TARGET="取引価格（総額）_log"
TARGET="単位面積あたりの取引価格_log"

#グラフ関連のクラス
class Multiple_Charts:
    def __init__(self,df,df_test=None):
        self.df=df
        self.df_test=df_test
    
    def hist0(self):
        # 数値型の列だけを選択する
        num_df = self.df.select_dtypes(include="number")
        # 列名をリストに変換する
        num_cols = num_df.columns.tolist()
        # リストの要素ごとにヒストグラムを作成する
        for col in num_cols:
            # 列名をタイトルとして表示する
            plt.title(col)
            # ヒストグラムを描画する
            plt.hist(self.df[col].dropna(), bins=10, edgecolor="black")
            # グラフを表示する
            plt.show()

    def hist1(self):
        for i,col_name in enumerate(['log_最寄駅：距離（分）', 'log_面積（㎡）', '建築年','建ぺい率（％）', '容積率（％）', '取引時点', '取引価格（総額）_log']):
            plt.hist(self.df[col_name])
            plt.title(col_name)
            plt.show()
    
    def year_price_test(self):
        plt.plot(self.df.index,self.df.values)
        plt.title("y")
        plt.xlabel("ds")
        plt.ylabel("y")

    def year_price(self):
        df_year_price=self.df.groupby("取引年")["取引価格（総額）_log"].mean()
        plt.plot(df_year_price.index,df_year_price.values)
        plt.title("y")
        plt.xlabel("ds")
        plt.ylabel("y")
        
    def year_MAE(self):
        df_year_price=self.df.groupby("取引年")["MAE"].mean()
        plt.plot(df_year_price.index,df_year_price.values)
        plt.title("y")
        plt.xlabel("ds")
        plt.ylabel("y")
    
    def distribution_map(self):
        train_pref=self.df["都道府県名"].value_counts(normalize=True)
        test_pref=self.df_test["都道府県名"].value_counts(normalize=True)
        plt.bar(train_pref.index,train_pref.values,label="train")
        plt.bar(test_pref.index,test_pref.values,label="test")
        plt.xticks(rotation=90)
        plt.xlabel("都道府県")
        plt.ylabel("頻度")
        plt.show()

        return train_pref,test_pref
    
    def graph_multiple_array(self):
        plt.plot(self.df)
        plt.show()

    def scatter_matrix_view(self):
        pd.plotting.scatter_matrix(self.df,figsize=(20,20))
        plt.show()

class Aggregator:
    
    def num_expr2(df, cols):
        # '都道府県名'でグループ化し、指定されたカラムの最大値、平均値、中央値を計算
        grouped_agg = df.groupby("都道府県名")[cols].agg([
                                                        "max",
                                                        "mean", 
                                                        "median",
                                                        "count",
                                                        #"var",
                                                        ])
        # カラム名を設定
        grouped_agg.columns = [
                            f"{cols}_max", 
                            f"{cols}_mean", 
                            f"{cols}_median",
                            f"{cols}_count",
                            #f"{cols}_var",
                            ]
        return grouped_agg

    def get_exprs(df,cols):
        # 上記のすべての集約関数を結合して返す
        exprs = Aggregator.num_expr(df,cols) 
        #print(exprs)

        return exprs


def read_df(df, group_col,exc_col,TARGET):
    #num_cols = list(df.select_dtypes(exclude=["object"]).columns)
    num_cols=[col for col in df.select_dtypes(exclude=["object"]).columns if col !=TARGET and col not in exc_col]
    agg_df = pd.DataFrame(index=df[group_col].unique())
    for num_col in num_cols:
        #group_df = df.groupby(group_col)[num_col].agg(Aggregator.get_exprs)
        group_df = df.groupby(group_col).apply(lambda x: Aggregator.num_expr2(x, num_col))
        #display(group_df)
        # 結果をデータフレームに変換
        group_df = group_df.reset_index(level=0, drop=True)
        #group_df=pd.DataFrame(group_df)
        agg_df = pd.concat([agg_df, group_df], axis=1)
        #print("-------test-------")
        #display(group_df)
        #agg_df = pd.merge(agg_df, group_df, on=group_col, how='outer')
        #agg_df=agg_df.join(group_df,how="outer")
    
    df=df.join(agg_df, on=group_col,how="left")
    return df

def to_cat_col(df_data,cat_cols=None):
    if cat_cols is None:# カテゴリ型に変換する列が指定されていない場合
        cat_cols = list(df_data.select_dtypes("object").columns)# オブジェクト型の列をすべて選択
    #df_data[cat_cols] = df_data[cat_cols].astype("category")# 指定された列をカテゴリ型に変換
    return df_data, cat_cols# 変換後のデータフレームとカテゴリ型の列のリストを返す

class VotingModel(BaseEstimator, RegressorMixin):
    def __init__(self, estimators,cat_cols):
        super().__init__()
        self.estimators = estimators
        self.cat_cols=cat_cols
        
    def fit(self, X, y=None):
        return self
    
    def predict_cat(self, X):
        X[self.cat_cols]=X[self.cat_cols].astype(str)
        y_preds = [estimator.predict(X) for estimator in self.estimators] # 各モデルの予測値を計算
        return np.mean(y_preds, axis=0)# 予測値の平均を返す
    
    def predict_lgbm(self,X):
        X[self.cat_cols]=X[self.cat_cols].astype("category")
        y_preds = [estimator.predict(X) for estimator in self.estimators] # 各モデルの予測値を計算
        return np.mean(y_preds, axis=0)# 予測値の平均を返す
    
    def predict_xg(self,X):
        X[self.cat_cols]=X[self.cat_cols].astype("category")
        y_preds = [estimator.predict(X) for estimator in self.estimators] # 各モデルの予測値を計算
        return np.mean(y_preds, axis=0)# 予測値の平均を返す
    
    def predict_proba(self, X):
        #y_preds = [estimator.predict_proba(X) for estimator in self.estimators]
        #y_preds = [estimator.predict_proba(X) for estimator in self.estimators[:5]]# 各モデルの予測確率を計算

        X[self.cat_cols]=X[self.cat_cols].astype("category")
        y_preds=[estimator.predict_proba(X) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)# 予測確率の平均を返す


class TreeExecution:
    def __init__(self,df,y_name,cat_cols):
        self.df=df
        self.y_name=y_name
        self.cat_cols=cat_cols
        self.df_copy=self.df
        self.df_compare=[]
        self.train=None
        self.test=None
        self.train_x=None
        self.train_y=None
        self.val_x=None
        self.val_y=None
        self.df_x=None
        self.df_y=None

        self.train_final_index=None
        self.cv_list=None

        self.rfr=None
        self.y_pred_rfr=None
        
        self.lightgbm_model=None
        self.y_pred=None

        self.catboost_model=None
        self.y_pred_catboost=None

        self.train_valid_x=None
        self.valid_x=None
        self.train_valid_y=None
        self.valid_y=None

    
    def lightgbm_simple(self):
        #X=self.df.drop(colmuns=[f"{self.y_name}"])
        #y=self.df[f"{self.y_name}"]

        self.train,self.test=train_test_split(self.df,stratify=self.df["取引時点"],test_size=0.12)
        print("self.df.shape",self.df.shape)
        print("self.train.shape",self.train.shape)
        print("self.test.shape",self.test.shape)

        self.train_x=self.train.drop(self.y_name,axis=1)
        self.train_y=self.train[self.y_name]
        self.val_x=self.test.drop(self.y_name,axis=1)
        self.val_y=self.test[self.y_name]

        self.train_x[self.cat_cols] = self.train_x[self.cat_cols].astype("category")
        self.val_x[self.cat_cols] = self.val_x[self.cat_cols].astype("category")

        params={"objective":"huber",
                "metrics":"mae",'n_estimators': 11694, 'early_stopping_round': 354, 'num_leaves': 923, 'max_depth': 18, 'feature_fraction': 0.5550780976534292, 'subsample_freq': 5, 'bagging_fraction': 0.9257381196775767, 'min_data_in_leaf': 22, 'lambda_l1': 0.009074027558328245, 'lambda_l2': 1.8570479012661552e-08}
        
        params1={
            "boosting_type": "gbdt",
            "objective": "l1",
            "metric": "mae",
            'max_depth': 47, 
            'learning_rate': 0.04597278751798127, 
            'num_leaves': 745, 
            'n_estimators': 29016, 
            'colsample_bytree': 0.20626653136275738, 
            'colsample_bynode': 0.11357016976115053, 
            'min_child_samples': 233, 
            'min_child_weight': 9.766535791974848, 
            'subsample': 0.27753502774657446, 
            'bagging_fraction': 0.9219003898291823, 
            'feature_fraction': 0.5723684151962363, 
            'min_split_gain': 0.523315793255011, 
            'max_bin': 8913, 
            'epsilon': 5.641899055323626, 
            'alpha': 0.9201577534062614, 
            'min_sum_hessian_in_leaf': 0.6820573797568824,
            "boost_from_average":False,
            #"early_stopping_rounds": 50,
            "verbose": -1,
            "random_state": 42
            }
        
        self.lightgbm_model=lgb.LGBMRegressor(**params1)
        self.lightgbm_model.fit(self.train_x,self.train_y,eval_set=(self.val_x,self.val_y),callbacks = [lgb.log_evaluation(200), lgb.early_stopping(100)])

        self.y_pred=self.lightgbm_model.predict(self.val_x)
        print(self.y_pred)
        print(mean_absolute_error(self.y_pred,self.val_y))

        return self.lightgbm_model

    def lightgbm_l1_cv(self):
        X=self.df.drop(columns=[f"{self.y_name}"])
        y=self.df[f"{self.y_name}"]
        prefecture_labels=X["都道府県名"]
        #threshold=0.074
        #cv=KFold(n_splits=3,shuffle=True)
        cv = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
        
        params={
        "boosting_type": "gbdt",
        #"class_weight": null,
        "colsample_bytree": 0.2661179203230455,
        "importance_type": "split",
        "learning_rate": 0.04804216261746864,
        "max_depth": 128,
        "min_child_samples": 577,
        "min_child_weight": 5.774925183375773,
        "min_split_gain": 0.3914873364805788,
        "n_estimators": 46022,#46022
        #"n_jobs": null,
        "num_leaves": 31,
        "objective": "l1",
        "random_state": 42,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "subsample": 0.528846990441364,
        "subsample_for_bin": 200000,
        "subsample_freq": 0,
        "metric": "mae",
        "colsample_bynode": 0.216985581379894,
        "bagging_fraction": 0.9760006064193019,
        "feature_fraction": 0.5336978769826026,
        "max_bin": 1123,
        "epsilon": 4.954717758504798,
        "alpha": 0.5725841971738328,
        "min_sum_hessian_in_leaf": 1.9378627484627984,
        "boost_from_average": False,
        "verbose": -1
    }
        #params={'max_depth': 128, 'learning_rate': 0.04804216261746864, 'n_estimators': 46022, 'colsample_bytree': 0.2661179203230455, 'colsample_bynode': 0.216985581379894, 'min_child_samples': 577, 'min_child_weight': 5.774925183375773, 'subsample': 0.528846990441364, 'bagging_fraction': 0.9760006064193019, 'feature_fraction': 0.5336978769826026, 'min_split_gain': 0.3914873364805788, 'max_bin': 1123, 'epsilon': 4.954717758504798, 'alpha': 0.5725841971738328, 'min_sum_hessian_in_leaf': 1.9378627484627984}

        params={'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 0.2661179203230455, 'importance_type': 'split', 'learning_rate': 0.04804216261746864, 'max_depth': 128, 'min_child_samples': 577, 'min_child_weight': 5.774925183375773, 'min_split_gain': 0.3914873364805788, 'n_estimators': 46022, 'n_jobs': None, 'num_leaves': 31, 'objective': 'l1', 'random_state': 42, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'subsample': 0.528846990441364, 'subsample_for_bin': 200000, 'subsample_freq': 0, 'metric': 'mae', 'colsample_bynode': 0.216985581379894, 'bagging_fraction': 0.9760006064193019, 'feature_fraction': 0.5336978769826026, 'max_bin': 1123, 'epsilon': 4.954717758504798, 'alpha': 0.5725841971738328, 'min_sum_hessian_in_leaf': 1.9378627484627984, 'boost_from_average': False, 'verbose': -1}

        fitted_models_cat = []
        fitted_models_lgbm = []
        cv_scores_cat = []
        cv_scores_lgbm = []
        oof_pred_cat=np.zeros(X.shape[0])
        oof_pred_lgbm=np.zeros(X.shape[0])


        for idx_train, idx_valid in cv.split(X, prefecture_labels
                                             #, groups=weeks
                                             ):#   Because it takes a long time to divide the data set, 
            X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]# each time the data set is divided, two models are trained to each other twice, which saves time.
            X_valid, y_valid = X.iloc[idx_valid], y.iloc[idx_valid]

            X_train[self.cat_cols] = X_train[self.cat_cols].astype("category")
            X_valid[self.cat_cols] = X_valid[self.cat_cols].astype("category")
            trains=lgb.Dataset(X_train,y_train)
            valids=lgb.Dataset(X_valid,y_valid)
            

            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set = (X_valid, y_valid),
                #sample_weight=train_weights,
                callbacks = [lgb.log_evaluation(200), lgb.early_stopping(100)] )
            
            y_pred_valid = model.predict(X_valid)
            oof_pred_lgbm[idx_valid]=y_pred_valid
            mae_score = mean_absolute_error(y_valid, y_pred_valid)

            fitted_models_lgbm.append(model)
            cv_scores_lgbm.append(mae_score)


            #shap.summary_plot(shap_values, X_valid)
            #shap.summary_plot(shap_values, X_valid, plot_type="bar")

            # Force Plot (最初のデータに対して)
            #shap.force_plot(explainer.expected_value, shap_values)

            # Waterfall Plot (最初のデータに対して)
            #shap.plots.waterfall(shap.Explanation(values=shap_values, base_values=explainer.expected_value))


        mean_mae_score=mean(cv_scores_lgbm)
        print("CV AUC scores: ", cv_scores_lgbm)
        print("Maximum CV AUC score: ", max(cv_scores_lgbm))
        print("Mean CV mae score: ", mean_mae_score)

        oof_df=self.df.copy()
        oof_df=pd.DataFrame(index=self.df.index.copy())
        oof_df["target"]=y
        #oof_df["pred_cat"]=oof_pred_cat
        oof_df["pred_lgbm"]=oof_pred_lgbm


        #-------------------
        #特徴量重要度
        for i in range(len(fitted_models_lgbm)):
            print(f"Fold-{i}:")
            lgb.plot_importance(fitted_models_lgbm[i], importance_type="gain", 
                                figsize=(10,12)
                                )
            plt.show()

            features = X_train.columns
            importances = fitted_models_lgbm[i].feature_importances_
            feature_importance = pd.DataFrame({'importance':importances,'features':features}).sort_values('importance', ascending=False).reset_index(drop=True)
            feature_importance

        #feature_importance.to_csv("特徴量重要度.csv")
        #-------------------

        return fitted_models_lgbm,mean_mae_score,oof_df
    
    def lightgbm_l1_cv_tg(self):
        X=self.df.drop(columns=[f"{self.y_name}"])
        y=self.df[f"{self.y_name}"]
        prefecture_labels=X["都道府県名"]
        #threshold=0.074
        #cv=KFold(n_splits=3,shuffle=True)
        cv = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
        
        params={
            "boosting_type": "gbdt","objective":"l1","metrics":"mae","boost_from_average":False,"verbose": -1,"random_state": 42,
                'max_depth': 146, 'learning_rate': 0.04475736155212619, 'n_estimators': 48260, 'colsample_bytree': 0.25115722950204683, 'colsample_bynode': 0.23761739079852606, 'min_child_samples': 594, 'min_child_weight': 8.279252981359091, 'subsample': 0.6115829632198598, 'bagging_fraction': 0.9163657118215358, 'feature_fraction': 0.5366140292624185, 'min_split_gain': 0.7386917462339693, 'max_bin': 1728, 'epsilon': 7.743499163314079, 'alpha': 0.7814548201134592, 'min_sum_hessian_in_leaf': 0.05058971217721564
            }
        params1={'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 0.25115722950204683, 'importance_type': 'split', 'learning_rate': 0.04475736155212619, 'max_depth': 146, 'min_child_samples': 594, 'min_child_weight': 8.279252981359091, 'min_split_gain': 0.7386917462339693, 'n_estimators': 48260, 'n_jobs': None, 'num_leaves': 31, 'objective': 'l1', 'random_state': 42, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'subsample': 0.6115829632198598, 'subsample_for_bin': 200000, 'subsample_freq': 0, 'metrics': 'mae', 'boost_from_average': False, 'verbose': -1, 'colsample_bynode': 0.23761739079852606, 'bagging_fraction': 0.9163657118215358, 'feature_fraction': 0.5366140292624185, 'max_bin': 1728, 'epsilon': 7.743499163314079, 'alpha': 0.7814548201134592, 'min_sum_hessian_in_leaf': 0.05058971217721564}

        fitted_models_cat = []
        fitted_models_lgbm = []
        cv_scores_cat = []
        cv_scores_lgbm = []
        oof_pred_cat=np.zeros(X.shape[0])
        oof_pred_lgbm=np.zeros(X.shape[0])


        for idx_train, idx_valid in cv.split(X, prefecture_labels
                                             #, groups=weeks
                                             ):#   Because it takes a long time to divide the data set, 
            X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]# each time the data set is divided, two models are trained to each other twice, which saves time.
            X_valid, y_valid = X.iloc[idx_valid], y.iloc[idx_valid]

            X_train[self.cat_cols] = X_train[self.cat_cols].astype("category")
            X_valid[self.cat_cols] = X_valid[self.cat_cols].astype("category")
            trains=lgb.Dataset(X_train,y_train)
            valids=lgb.Dataset(X_valid,y_valid)
            

            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set = (X_valid, y_valid),
                #sample_weight=train_weights,
                callbacks = [lgb.log_evaluation(200), lgb.early_stopping(100)] )
            
            y_pred_valid = model.predict(X_valid)
            oof_pred_lgbm[idx_valid]=y_pred_valid
            mae_score = mean_absolute_error(y_valid, y_pred_valid)

            fitted_models_lgbm.append(model)
            cv_scores_lgbm.append(mae_score)

            #shap.summary_plot(shap_values, X_valid)
            #shap.summary_plot(shap_values, X_valid, plot_type="bar")

            # Force Plot (最初のデータに対して)
            #shap.force_plot(explainer.expected_value, shap_values)

            # Waterfall Plot (最初のデータに対して)
            #shap.plots.waterfall(shap.Explanation(values=shap_values, base_values=explainer.expected_value))


        mean_mae_score=mean(cv_scores_lgbm)
        print("CV AUC scores: ", cv_scores_lgbm)
        print("Maximum CV AUC score: ", max(cv_scores_lgbm))
        print("Mean CV mae score: ", mean_mae_score)

        oof_df=self.df.copy()
        oof_df=pd.DataFrame(index=self.df.index.copy())
        oof_df["target"]=y
        #oof_df["pred_cat"]=oof_pred_cat
        oof_df["pred_lgbm"]=oof_pred_lgbm


        #-------------------
        #特徴量重要度
        for i in range(len(fitted_models_lgbm)):
            print(f"Fold-{i}:")
            lgb.plot_importance(fitted_models_lgbm[i], importance_type="gain", 
                                figsize=(10,12)
                                )
            plt.show()

            features = X_train.columns
            importances = fitted_models_lgbm[i].feature_importances_
            feature_importance = pd.DataFrame({'importance':importances,'features':features}).sort_values('importance', ascending=False).reset_index(drop=True)
            feature_importance

        #feature_importance.to_csv("特徴量重要度.csv")
        #-------------------

        return fitted_models_lgbm,mean_mae_score,oof_df
    
    def lightgbm(self):
        X=self.df.drop(columns=[f"{self.y_name}"])
        y=self.df[f"{self.y_name}"]
        prefecture_labels=X["都道府県名"]
        #threshold=0.074
        #cv=KFold(n_splits=3,shuffle=True)
        cv = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
        
        params={
        "boosting_type": "gbdt",
        #"class_weight": null,
        "colsample_bytree": 0.2661179203230455,
        "importance_type": "split",
        "learning_rate": 0.04804216261746864,
        "max_depth": 128,
        "min_child_samples": 577,
        "min_child_weight": 5.774925183375773,
        "min_split_gain": 0.3914873364805788,
        "n_estimators": 46022,
        #"n_jobs": null,
        "num_leaves": 31,
        "objective": "l1",
        "random_state": 42,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "subsample": 0.528846990441364,
        "subsample_for_bin": 200000,
        "subsample_freq": 0,
        "metric": "mae",
        "colsample_bynode": 0.216985581379894,
        "bagging_fraction": 0.9760006064193019,
        "feature_fraction": 0.5336978769826026,
        "max_bin": 1123,
        "epsilon": 4.954717758504798,
        "alpha": 0.5725841971738328,
        "min_sum_hessian_in_leaf": 1.9378627484627984,
        "boost_from_average": False,
        "verbose": -1
    }

        fitted_models_cat = []
        fitted_models_lgbm = []
        cv_scores_cat = []
        cv_scores_lgbm = []
        oof_pred_cat=np.zeros(X.shape[0])
        oof_pred_lgbm=np.zeros(X.shape[0])


        for idx_train, idx_valid in cv.split(X, prefecture_labels
                                             #, groups=weeks
                                             ):#   Because it takes a long time to divide the data set, 
            X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]# each time the data set is divided, two models are trained to each other twice, which saves time.
            X_valid, y_valid = X.iloc[idx_valid], y.iloc[idx_valid]

            X_train[self.cat_cols] = X_train[self.cat_cols].astype("category")
            X_valid[self.cat_cols] = X_valid[self.cat_cols].astype("category")
            trains=lgb.Dataset(X_train,y_train)
            valids=lgb.Dataset(X_valid,y_valid)
            

            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set = (X_valid, y_valid),
                #sample_weight=train_weights,
                callbacks = [lgb.log_evaluation(200), lgb.early_stopping(100)] )
            
            y_pred_valid = model.predict(X_valid)
            oof_pred_lgbm[idx_valid]=y_pred_valid
            mae_score = mean_absolute_error(y_valid, y_pred_valid)

            fitted_models_lgbm.append(model)
            cv_scores_lgbm.append(mae_score)

        mean_mae_score=mean(cv_scores_lgbm)
        print("CV AUC scores: ", cv_scores_lgbm)
        print("Maximum CV AUC score: ", max(cv_scores_lgbm))
        print("Mean CV mae score: ", mean_mae_score)

        oof_df=self.df.copy()
        oof_df=pd.DataFrame(index=self.df.index.copy())
        oof_df["target"]=y
        #oof_df["pred_cat"]=oof_pred_cat
        oof_df["pred_lgbm"]=oof_pred_lgbm


        #-------------------
        #特徴量重要度
        for i in range(len(fitted_models_lgbm)):
            print(f"Fold-{i}:")
            lgb.plot_importance(fitted_models_lgbm[i], importance_type="gain", 
                                figsize=(10,10)
                                )
            plt.show()

            features = X_train.columns
            importances = fitted_models_lgbm[i].feature_importances_
            feature_importance = pd.DataFrame({'importance':importances,'features':features}).sort_values('importance', ascending=False).reset_index(drop=True)
            feature_importance

        #feature_importance.to_csv("特徴量重要度.csv")
        #-------------------

        return fitted_models_lgbm,mean_mae_score,oof_df
    
    def lightgbm_l1_cv_optuna_holdout(self):
        X=self.df.drop(columns=[f"{self.y_name}"])
        y=self.df[f"{self.y_name}"]
        prefecture_labels=X["都道府県名"]
        #threshold=0.074
        #cv=KFold(n_splits=3,shuffle=True)
        self.train,self.test=train_test_split(self.df,stratify=self.df["都道府県名"],test_size=0.12,random_state=42)
        print("self.df.shape",self.df.shape)
        print("self.train.shape",self.train.shape)
        print("self.test.shape",self.test.shape)

        self.train_x=self.train.drop(self.y_name,axis=1)
        self.train_y=self.train[self.y_name]
        self.val_x=self.test.drop(self.y_name,axis=1)
        self.val_y=self.test[self.y_name]

        #LightGBMのカスタム損失関数
        def lq_loss_1(q):
            def loss(y_pred, data):
                y_true = data.get_label()
                # gradは、損失関数の一階微分（勾配）を計算する
                # np.sign(y_pred - y_true)は、予測値と実際の値の差の符号(-1,0,1のどれか)を取得
                # np.abs(y_pred - y_true) ** (q - 1)は、差の絶対値を(q - 1)乗する
                grad = np.sign(y_pred - y_true) * np.abs(y_pred - y_true) ** (q - 1)
                # hessは、損失関数の二階微分（ヘッシアン）を計算する
                # (q - 1) * np.abs(y_pred - y_true) ** (q - 2)は、差の絶対値を(q - 2)乗し、(q - 1)を掛け
                hess = (q - 1) * np.abs(y_pred - y_true) ** (q - 2)
                return grad, hess
            return loss

        def lq_loss(y_true,y_pred,q=1.0):
            # gradは、損失関数の一階微分（勾配）を計算する
            # np.sign(y_pred - y_true)は、予測値と実際の値の差の符号(-1,0,1のどれか)を取得
            # np.abs(y_pred - y_true) ** (q - 1)は、差の絶対値を(q - 1)乗する
            grad = np.sign(y_pred - y_true) * np.abs(y_pred - y_true) ** (q - 1)
            # hessは、損失関数の二階微分（ヘッシアン）を計算する
            # (q - 1) * np.abs(y_pred - y_true) ** (q - 2)は、差の絶対値を(q - 2)乗し、(q - 1)を掛け
            hess = (q - 1) * np.abs(y_pred - y_true) ** (q - 2)
            return grad, hess


        def objective(trial):

            q = trial.suggest_float("q", 1.0, 2.0)

            params2 = {
            "boosting_type": "gbdt",
            #'objective': 'huber',
            'objective': lambda y_true,y_pred: lq_loss(y_true,y_pred,q),
            "metric": "mae",
            "num_leaves": trial.suggest_int("num_leaves", 2, 25000),
            "max_depth": trial.suggest_int("max_depth", 5, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.05),
            "n_estimators": trial.suggest_int("n_estimators", 1500, 50000),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1),#0.5
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.1, 1),#0.5
            "min_child_samples": trial.suggest_int("min_child_samples", 100, 1000),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 10),
            "subsample": trial.suggest_float("subsample", 0.2, 1),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.01, 1),
            "max_bin": trial.suggest_int("max_bin", 500, 10000),
            "epsilon": trial.suggest_float("epsilon", 1.0, 10.0),  # Huber損失関数のパラメータを追加
            "alpha":trial.suggest_float("alpha",0.01,0.99),
            "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", 1e-3, 10.0),  # 追加
            "boost_from_average":False,
            #"early_stopping_rounds": 50,
            "verbose": -1,
            "random_state": 42
            }

            self.train_x[self.cat_cols] = self.train_x[self.cat_cols].astype("category")
            self.val_x[self.cat_cols] = self.val_x[self.cat_cols].astype("category")

            # データセットの作成
            #train_data = lgb.Dataset(self.train_x, label=self.train_y)
            #val_data = lgb.Dataset(self.val_x, label=self.val_y, reference=train_data)

            self.lightgbm_model=lgb.LGBMRegressor(**params2)
            self.lightgbm_model.fit(
                                    self.train_x,self.train_y,
                                    eval_set=[(self.val_x,self.val_y)],
                                    #train_data,val_data,
                                    #eval_set=[val_data],
                                    callbacks = [lgb.log_evaluation(1250), lgb.early_stopping(100)])
            y_pred_valid=self.lightgbm_model.predict(self.val_x)
            mae_score=mean_absolute_error(self.val_y,y_pred_valid)

            # 最良モデルを保存
            # トライアルがプルーニング（早期終了）されるべきかをチェック
            if trial.should_prune():
                print("トライアルがプルーニング（早期終了）されました")
                raise optuna.TrialPruned()  # プルーニングされる場合、例外を投げてトライアルを終了

            # 現在のトライアルのスコアがこれまでの最良スコアよりも良いかをチェック
            if not hasattr(objective, "best_score") or mae_score < objective.best_score:
                # 最良スコアと最良モデルを更新
                objective.best_score = mae_score  # 最良スコアを更新
                objective.best_model = self.lightgbm_model  # 最良モデルを更新
                print(f"最良スコアを更新しました -> {mae_score}")


            gc.collect()  # ガベージコレクションを呼び出す
            trial.set_user_attr("models",self.lightgbm_model)
            print("MAE scores: ", mae_score)

            return mae_score

        sampler = optuna.samplers.TPESampler(multivariate=True)
        study=optuna.create_study(direction="minimize",sampler=sampler)
        #study.enqueue_trial({#'q':1.0,'max_depth': 47,'learning_rate': 0.04597278751798127,'n_estimators': 29016,'colsample_bytree': 0.20626653136275738,'colsample_bynode': 0.11357016976115053,'min_child_samples': 233,'min_child_weight': 9.766535791974848,'min_split_gain': 0.523315793255011,'subsample': 0.27753502774657446,'bagging_fraction': 0.9219003898291823,'feature_fraction': 0.5723684151962363,'max_bin': 8913,'epsilon': 5.641899055323626,'alpha': 0.9201577534062614,'min_sum_hessian_in_leaf': 0.6820573797568824,})
        study.enqueue_trial({'q': 1.4547403725924728, 'num_leaves': 20, 'max_depth': 84, 'learning_rate': 0.033545467220629066, 'n_estimators': 12083, 'colsample_bytree': 0.23947776719856848, 'colsample_bynode': 0.22255315097411035, 'min_child_samples': 358, 'min_child_weight': 1.9424886962107888, 'subsample': 0.4278296849323956, 'bagging_fraction': 0.9687803233331815, 'feature_fraction': 0.7779705049152938, 'min_split_gain': 0.3661721169413612, 'max_bin': 7177, 'epsilon': 1.7896297256997062, 'alpha': 0.8344086668989001, 'min_sum_hessian_in_leaf': 4.258607129429245})
        #0.06685978690445642
        study.optimize(objective,n_trials=120,timeout=72000)

        # 最良モデルを取得
        model_lgbm = objective.best_model
        #model_cat=study.best_trial.user_attrs["models"]
        mae_score=study.best_value

        lgb.plot_importance(model_lgbm, importance_type="gain",
                                figsize=(20,20)
                                )
        plt.show()

        features = self.train_x.columns
        importances = model_lgbm.feature_importances_
        feature_importance = pd.DataFrame({'importance':importances,'features':features}).sort_values('importance', ascending=False).reset_index(drop=True)
        feature_importance

        return model_lgbm,mae_score
    
    def lightgbm_l1_cv_tg_optuna_holdout(self):
        X=self.df.drop(columns=[f"{self.y_name}"])
        y=self.df[f"{self.y_name}"]
        prefecture_labels=X["都道府県名"]
        #threshold=0.074
        #cv=KFold(n_splits=3,shuffle=True)
        self.train,self.test=train_test_split(self.df,stratify=self.df["都道府県名"],test_size=0.12,random_state=42)
        print("self.df.shape",self.df.shape)
        print("self.train.shape",self.train.shape)
        print("self.test.shape",self.test.shape)

        self.train_x=self.train.drop(self.y_name,axis=1)
        self.train_y=self.train[self.y_name]
        self.val_x=self.test.drop(self.y_name,axis=1)
        self.val_y=self.test[self.y_name]

        #LightGBMのカスタム損失関数
        def lq_loss_1(q):
            def loss(y_pred, data):
                y_true = data.get_label()
                # gradは、損失関数の一階微分（勾配）を計算する
                # np.sign(y_pred - y_true)は、予測値と実際の値の差の符号(-1,0,1のどれか)を取得
                # np.abs(y_pred - y_true) ** (q - 1)は、差の絶対値を(q - 1)乗する
                grad = np.sign(y_pred - y_true) * np.abs(y_pred - y_true) ** (q - 1)
                # hessは、損失関数の二階微分（ヘッシアン）を計算する
                # (q - 1) * np.abs(y_pred - y_true) ** (q - 2)は、差の絶対値を(q - 2)乗し、(q - 1)を掛け
                hess = (q - 1) * np.abs(y_pred - y_true) ** (q - 2)
                return grad, hess
            return loss

        def lq_loss(y_true,y_pred,q=1.0):
            # gradは、損失関数の一階微分（勾配）を計算する
            # np.sign(y_pred - y_true)は、予測値と実際の値の差の符号(-1,0,1のどれか)を取得
            # np.abs(y_pred - y_true) ** (q - 1)は、差の絶対値を(q - 1)乗する
            grad = np.sign(y_pred - y_true) * np.abs(y_pred - y_true) ** (q - 1)
            # hessは、損失関数の二階微分（ヘッシアン）を計算する
            # (q - 1) * np.abs(y_pred - y_true) ** (q - 2)は、差の絶対値を(q - 2)乗し、(q - 1)を掛け
            hess = (q - 1) * np.abs(y_pred - y_true) ** (q - 2)
            return grad, hess


        def objective(trial):

            #q = trial.suggest_float("q", 1.0, 2.0)

            params2 = {
            "boosting_type": "gbdt",
            'objective': 'l1',
            #'objective': lambda y_true,y_pred: lq_loss(y_true,y_pred,q),
            "metric": "mae",
            "max_depth": trial.suggest_int("max_depth", 5, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.05),
            "n_estimators": trial.suggest_int("n_estimators", 1500, 50000),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1),#0.5
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.1, 1),#0.5
            "min_child_samples": trial.suggest_int("min_child_samples", 100, 1000),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 10),
            "subsample": trial.suggest_float("subsample", 0.2, 1),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.0, 1),#0.5
            "feature_fraction": trial.suggest_float("feature_fraction", 0.0, 1),#0.5
            "min_split_gain": trial.suggest_float("min_split_gain", 0.01, 1),
            "max_bin": trial.suggest_int("max_bin", 500, 10000),
            "epsilon": trial.suggest_float("epsilon", 1.0, 10.0),  # Huber損失関数のパラメータを追加
            "alpha":trial.suggest_float("alpha",0.01,0.99),
            "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", 1e-3, 10.0),  # 追加
            "boost_from_average":False,
            #"early_stopping_rounds": 50,
            "verbose": -1,
            "random_state": 42
            }

            self.train_x[self.cat_cols] = self.train_x[self.cat_cols].astype("category")
            self.val_x[self.cat_cols] = self.val_x[self.cat_cols].astype("category")

            # データセットの作成
            #train_data = lgb.Dataset(self.train_x, label=self.train_y)
            #val_data = lgb.Dataset(self.val_x, label=self.val_y, reference=train_data)

            self.lightgbm_model=lgb.LGBMRegressor(**params2)
            self.lightgbm_model.fit(
                                    self.train_x,self.train_y,
                                    eval_set=[(self.val_x,self.val_y)],
                                    #train_data,val_data,
                                    #eval_set=[val_data],
                                    callbacks = [lgb.log_evaluation(1250), lgb.early_stopping(100)])
            y_pred_valid=self.lightgbm_model.predict(self.val_x)
            mae_score=mean_absolute_error(self.val_y,y_pred_valid)

            # 最良モデルを保存
            # トライアルがプルーニング（早期終了）されるべきかをチェック
            if trial.should_prune():
                print("トライアルがプルーニング（早期終了）されました")
                raise optuna.TrialPruned()  # プルーニングされる場合、例外を投げてトライアルを終了

            # 現在のトライアルのスコアがこれまでの最良スコアよりも良いかをチェック
            if not hasattr(objective, "best_score") or mae_score < objective.best_score:
                # 最良スコアと最良モデルを更新
                objective.best_score = mae_score  # 最良スコアを更新
                objective.best_model = self.lightgbm_model  # 最良モデルを更新
                print(f"最良スコアを更新しました -> {mae_score}")


            gc.collect()  # ガベージコレクションを呼び出す
            trial.set_user_attr("models",self.lightgbm_model)
            print("MAE scores: ", mae_score)

            return mae_score

        sampler = optuna.samplers.TPESampler(multivariate=True)
        study=optuna.create_study(direction="minimize",sampler=sampler)
        study.enqueue_trial({'max_depth': 128, 'learning_rate': 0.04804216261746864, 'n_estimators': 46022, 'colsample_bytree': 0.2661179203230455, 'colsample_bynode': 0.216985581379894, 'min_child_samples': 577, 'min_child_weight': 5.774925183375773, 'subsample': 0.528846990441364, 'bagging_fraction': 0.9760006064193019, 'feature_fraction': 0.5336978769826026, 'min_split_gain': 0.3914873364805788, 'max_bin': 1123, 'epsilon': 4.954717758504798, 'alpha': 0.5725841971738328, 'min_sum_hessian_in_leaf': 1.9378627484627984})
        study.enqueue_trial({
                            #'q':1.0,
                            'max_depth': 47,
                            'learning_rate': 0.04597278751798127,
                            'n_estimators': 29016,
                            'colsample_bytree': 0.20626653136275738,
                            'colsample_bynode': 0.11357016976115053,
                            'min_child_samples': 233,
                            'min_child_weight': 9.766535791974848,
                            'min_split_gain': 0.523315793255011,
                            'subsample': 0.27753502774657446,
                            'bagging_fraction': 0.9219003898291823,
                            'feature_fraction': 0.5723684151962363,
                            'max_bin': 8913,
                            'epsilon': 5.641899055323626,
                            'alpha': 0.9201577534062614,
                            'min_sum_hessian_in_leaf': 0.6820573797568824,
                            })
        #study.enqueue_trial({'max_depth': 32, 'learning_rate': 0.03906871276163297, 'n_estimators': 36544, 'colsample_bytree': 0.5126730998221444, 'colsample_bynode': 0.2025285932597008, 'min_child_samples': 157, 'min_child_weight': 5.3273927589644465, 'subsample': 0.5319492213033499, 'bagging_fraction': 0.8816841567606699, 'feature_fraction': 0.5095242100115367, 'min_split_gain': 0.519663274547186, 'max_bin': 8327, 'epsilon': 2.1318409326401726, 'alpha': 0.9217865814989281, 'min_sum_hessian_in_leaf': 3.231420423646926})
        #study.enqueue_trial({'max_depth': 128, 'learning_rate': 0.04804216261746864, 'n_estimators': 46022, 'colsample_bytree': 0.2661179203230455, 'colsample_bynode': 0.216985581379894, 'min_child_samples': 577, 'min_child_weight': 5.774925183375773, 'subsample': 0.528846990441364, 'bagging_fraction': 0.9760006064193019, 'feature_fraction': 0.5336978769826026, 'min_split_gain': 0.3914873364805788, 'max_bin': 1123, 'epsilon': 4.954717758504798, 'alpha': 0.5725841971738328, 'min_sum_hessian_in_leaf': 1.9378627484627984})
        study.optimize(objective,n_trials=120,timeout=72000)

        # 最良モデルを取得
        model_lgbm = objective.best_model
        #model_cat=study.best_trial.user_attrs["models"]
        mae_score=study.best_value

        lgb.plot_importance(model_lgbm, importance_type="gain",
                                figsize=(20,20)
                                )
        plt.show()

        features = self.train_x.columns
        importances = model_lgbm.feature_importances_
        feature_importance = pd.DataFrame({'importance':importances,'features':features}).sort_values('importance', ascending=False).reset_index(drop=True)
        feature_importance

        return model_lgbm,mae_score


    def catboost(self):
        self.df[self.cat_cols] = self.df[self.cat_cols].astype(str)

        X=self.df.drop(columns=[f"{self.y_name}"])
        y=self.df[f"{self.y_name}"]
        prefecture_labels=X["都道府県名"]
        #threshold=0.073
        cv = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)


        params_cat2={
                    'iterations': 16321,#16321
                    'learning_rate': 0.02141285001219377,
                    'depth': 12,
                    'eval_metric': 'MAE',  # 評価指標
                    'od_type': 'Iter',  # 過学習検出のタイプ
                    "use_best_model":True,
                    'l2_leaf_reg': 1.1080618593367644,
                    'border_count': 226,
                    'min_data_in_leaf': 14,
                    'od_wait': 448,
                    'best_model_min_trees': 2191,
                    'loss_function': 'Lq:q=1.3631413553264038',
                    'random_strength': 8.902514382937806,
                    'leaf_estimation_iterations': 7,
                    'leaf_estimation_backtracking': 'AnyImprovement',
                    'score_function': 'NewtonCosine',
                    #'random_seed':2
                    }

        fitted_models_cat = []
        fitted_models_lgbm = []
        cv_scores_cat = []
        cv_scores_lgbm = []
        oof_pred_cat=np.zeros(X.shape[0])
        oof_pred_lgbm=np.zeros(X.shape[0])
        fold=1


        for idx_train, idx_valid in cv.split(X, prefecture_labels):#   Because it takes a long time to divide the data set,
            X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]# each time the data set is divided, two models are trained to each other twice, which saves time.
            X_valid, y_valid = X.iloc[idx_valid], y.iloc[idx_valid]
            train_pool=Pool(X_train,y_train,cat_features=self.cat_cols)
            val_pool=Pool(X_valid,y_valid,cat_features=self.cat_cols)
            clf=CatBoostRegressor(**params_cat2,task_type="GPU",verbose=False)

            clf.fit(train_pool,eval_set=val_pool,verbose=500,early_stopping_rounds=150)#500
            fitted_models_cat.append(clf)
            y_pred_valid=clf.predict(X_valid)
            oof_pred_cat[idx_valid]=y_pred_valid
            mae_score=mean_absolute_error(y_valid,y_pred_valid)
            cv_scores_cat.append(mae_score)
            print("MAE_score: ",mae_score)

            # 各foldのモデルを保存
            clf.save_model(f'/content/drive/MyDrive/yuto/project_directory/models/v2_model_cat_lq_fold_{fold}.cbm', format='cbm')
            print(f"modelの保存 -> fold{fold}終了")
            fold += 1

            #shap.summary_plot(shap_values, X_valid)
            #shap.summary_plot(shap_values, X_valid, plot_type="bar")

            # Force Plot (最初のデータに対して)
            #shap.force_plot(explainer.expected_value, shap_values)

            # Waterfall Plot (最初のデータに対して)
            #shap.plots.waterfall(shap.Explanation(values=shap_values, base_values=explainer.expected_value))


        mean_mae_score=mean(cv_scores_cat)
        print("CV AUC scores: ", cv_scores_cat)
        print("Maximum CV AUC score: ", max(cv_scores_cat))
        print("Mean CV AUC score: ", mean_mae_score)

        #print("CV AUC scores: ", cv_scores_lgbm)
        #print("Maximum CV AUC score: ", max(cv_scores_lgbm))

        oof_df=pd.DataFrame(index=self.df.index.copy())
        oof_df["target"]=y
        oof_df["pred_cat"]=oof_pred_cat
        #oof_df["pred_lgbm"]=oof_pred_lgbm

        feature_importances_cat=[]
        #-------------------
        #特徴量重要度
        for i in range(len(fitted_models_cat)):
            print(f"Fold-{i}")
            feature_importances = fitted_models_cat[i].get_feature_importance(type='FeatureImportance')

            # 特徴量の重要度をDataFrameに変換
            features = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

            # 特徴量の重要度を降順にソート
            features = features.sort_values(by='Importance', ascending=False)
            feature_importances_cat.append(features)

            # 可視化
            plt.figure(figsize=(10, 14))
            sns.barplot(x='Importance', y='Feature', data=features)
            plt.title('Feature Importance')
            plt.tight_layout()
            plt.show()
        #-------------------

        return fitted_models_cat,mean_mae_score,oof_df
    
    def catboost_optuna_holdout(self):
        self.df[self.cat_cols] = self.df[self.cat_cols].astype(str)

        X=self.df.drop(columns=[f"{self.y_name}"])
        y=self.df[f"{self.y_name}"]
        prefecture_labels=X["都道府県名"]
        #threshold=0.074
        #cv=KFold(n_splits=3,shuffle=True)
        self.train,self.test=train_test_split(self.df,stratify=self.df["都道府県名"],test_size=0.125,random_state=42)
        print("self.df.shape",self.df.shape)
        print("self.train.shape",self.train.shape)
        print("self.test.shape",self.test.shape)

        self.train_x=self.train.drop(self.y_name,axis=1)
        self.train_y=self.train[self.y_name]
        self.val_x=self.test.drop(self.y_name,axis=1)
        self.val_y=self.test[self.y_name]

        # 不要になった変数を削除
        del self.train
        del self.test
        gc.collect()  # ガベージコレクションを呼び出す

        def objective(trial):


            params = {
                "iterations": trial.suggest_int("iterations", 100, 17500),  # イテレーション数
                "learning_rate": trial.suggest_uniform("learning_rate", 0.001, 0.1),  # 学習率
                "depth": trial.suggest_int("depth", 2, 16),  # 決定木の深さ
                "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1e-8, 10.0),  # L2正則化
                "border_count": trial.suggest_int("border_count", 1, 255),  # 分割数
                # "ctr_border_count": trial.suggest_int("ctr_border_count", 1, 255),  # カテゴリ特徴量の分割数（コメントアウト）
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 20),  # リーフに含まれる最小データ数
                "eval_metric": 'MAE',  # 評価指標
                "od_type": 'Iter',  # 過学習検出のタイプ
                # "grow_policy": 'Lossguide',  # ツリーの成長ポリシー（コメントアウト）
                "od_wait": trial.suggest_int("od_wait", 1, 500),  # 過学習検出の待機イテレーション数
                "use_best_model":True,#20240825ON
                "best_model_min_trees": trial.suggest_int("best_model_min_trees", 10, 2500),  # 最良モデルの最小ツリー数
                # "max_leaves": trial.suggest_int("max_leaves", 6, 124),  # 最大リーフ数（コメントアウト）
                # "loss_function": 'Lq:q=1.3',  # 損失関数（コメントアウト）
                "loss_function": f'Lq:q={trial.suggest_float("q", 1.0, 2.0)}',  # 損失関数（動的設定）
                #"bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS", "Poisson", "No"]),  # ブートストラップタイプ
                #"colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 1.0),  # 各レベルで使用する特徴量の割合(※rsmパラメータ有の扱いになるので注意)
                "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),  # ランダム強度
                "leaf_estimation_iterations": trial.suggest_int("leaf_estimation_iterations", 1, 10),  # リーフ推定の反復回数
                #"leaf_estimation_method": trial.suggest_categorical("leaf_estimation_method", ["Newton", "Gradient"]),  # リーフ推定方法
                "leaf_estimation_backtracking": trial.suggest_categorical("leaf_estimation_backtracking", ["AnyImprovement", "Armijo"]),
                #"boosting_type": trial.suggest_categorical("boosting_type", ["Plain", "Ordered"]),
                "score_function": trial.suggest_categorical("score_function", ["Cosine", "L2", "NewtonL2", "NewtonCosine"])
                # "random_seed": trial.suggest_int("random_seed", 1, 10000),  # 乱数シード（コメントアウト）
                # "random_state": trial.suggest_int("random_state", 1, 42)  # 乱数シード（コメントアウト）
                #"random_state": 42
                }

            #print(params)

            #if params["bootstrap_type"] == "Bayesian":
            #    params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)  # ベイズブートストラップの温度
            #elif params["bootstrap_type"] == "Bernoulli":
            #    params["subsample"] = trial.suggest_float("subsample", 0.1, 1)  # サブサンプルの割合


            #self.train_x[self.cat_cols] = self.train_x[self.cat_cols].astype("category")
            #self.val_x[self.cat_cols] = self.val_x[self.cat_cols].astype("category")

            train_pool=Pool(self.train_x,self.train_y,cat_features=self.cat_cols)
            val_pool=Pool(self.val_x,self.val_y,cat_features=self.cat_cols)
            clf=CatBoostRegressor(**params,task_type="GPU",verbose=False)
            #clf=CatBoostRegressor(iterations=1000,
            #                 learning_rate=0.05,
            #                 depth=10,
            #                 eval_metric='MAE',
            #                 od_type='Iter',
            #                 od_wait=200,
            #                 loss_function='Lq:q=1.3',
            #                 task_type="GPU",
            #                 devices="0:1"
            #                 )
            clf.fit(train_pool,eval_set=val_pool,verbose=500,early_stopping_rounds=150)
            y_pred_valid=clf.predict(self.val_x)
            mae_score=mean_absolute_error(self.val_y,y_pred_valid)
            #trial.set_user_attr("models",clf)
            print("MAE scores: ", mae_score)

            # 最良モデルを保存
            # トライアルがプルーニング（早期終了）されるべきかをチェック
            if trial.should_prune():
                print("トライアルがプルーニング（早期終了）されました")
                raise optuna.TrialPruned()  # プルーニングされる場合、例外を投げてトライアルを終了

            # 現在のトライアルのスコアがこれまでの最良スコアよりも良いかをチェック
            if not hasattr(objective, "best_score") or mae_score < objective.best_score:
                # 最良スコアと最良モデルを更新
                objective.best_score = mae_score  # 最良スコアを更新
                objective.best_model = clf  # 最良モデルを更新
                print(f"最良スコアを更新しました -> {mae_score}")

            gc.collect()  # ガベージコレクションを呼び出す

            return mae_score

        sampler = optuna.samplers.TPESampler(multivariate=True)
        pruner = optuna.pruners.MedianPruner()
        study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
        study.enqueue_trial({
                        'iterations': 16321,
                        'learning_rate': 0.02141285001219377,
                        'depth': 12,
                        'l2_leaf_reg': 1.1080618593367644,
                        'border_count': 226,
                        'min_data_in_leaf': 14,
                        "eval_metric": 'MAE',  # 評価指標
                        "od_type": 'Iter',  # 過学習検出のタイプ
                        'od_wait': 448,
                        "use_best_model":True,
                        'best_model_min_trees': 2191,
                        'q': 1.3631413553264038,
                        'random_strength': 8.902514382937806,
                        'leaf_estimation_iterations': 7,
                        'leaf_estimation_backtracking': 'AnyImprovement',
                        'score_function': 'NewtonCosine'
                        })
        study.enqueue_trial({
                        'iterations': 15569,
                        'learning_rate': 0.01930228155709196,
                        'depth': 11,
                        'l2_leaf_reg': 1.6298736389002677,
                        'border_count': 141,
                        'min_data_in_leaf': 18,
                        "eval_metric": 'MAE',  # 評価指標
                        "od_type": 'Iter',  # 過学習検出のタイプ
                        'od_wait': 483,
                        "use_best_model":True,
                        'best_model_min_trees': 2361,
                        'q': 1.2532776511622248,
                        'random_strength': 8.788677335701141,
                        'leaf_estimation_iterations': 7,
                        'leaf_estimation_backtracking': 'AnyImprovement',
                        'score_function': 'NewtonCosine'
                        })
        study.optimize(objective,n_trials=100,timeout=72000)

        # 最良モデルを取得
        model_cat = objective.best_model
        #model_cat=study.best_trial.user_attrs["models"]
        mae_score=study.best_value

        #-------------------
        #特徴量重要度
        feature_importances = model_cat.get_feature_importance(type='FeatureImportance')

        # 特徴量の重要度をDataFrameに変換
        features = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

        # 特徴量の重要度を降順にソート
        features = features.sort_values(by='Importance', ascending=False)

        # 可視化
        plt.figure(figsize=(8, 20))
        sns.barplot(x='Importance', y='Feature', data=features)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.show()
        #-------------------

        return model_cat,mae_score


    def catboost_optuna(self):
        self.df[self.cat_cols] = self.df[self.cat_cols].astype(str)

        X=self.df.drop(columns=[f"{self.y_name}"])
        y=self.df[f"{self.y_name}"]
        prefecture_labels=X["都道府県名"]
        #threshold=0.073
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        def objective(trial):
            params_cat={
                "eval_metric":"mae",
                "learning_rate":0.03,
                "iterations":300#6000
                    }
            
            params_cat2={"iterations":trial.suggest_int("iterations",1000,6000),
                        "learning_rate":trial.suggest_uniform("learning_rate",0.001,0.07),
                        "depth":trial.suggest_int("depth",2,16),
                        "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1e-8, 10.0),
                        "border_count": trial.suggest_int("border_count", 1, 255),
                        #"ctr_border_count": trial.suggest_int("ctr_border_count", 1, 255),
                        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 20),
                        "eval_metric":'MAE',
                        "od_type":'Iter',
                        #"grow_policy":'Lossguide',
                        "od_wait":trial.suggest_int("od_wait",200,500),
                        "best_model_min_trees":trial.suggest_int("best_model_min_trees",10,2500),
                        #"max_leaves":trial.suggest_int("max_leaves",6,124),
                        #"loss_function":'Lq:q=1.3',
                        "loss_function": f'Lq:q={trial.suggest_float("q", 1.0, 3.0)}'
                        #"random_state": trial.suggest_int("random_state", 1, 42)
                        }

            fitted_models_cat = []
            fitted_models_lgbm = []
            cv_scores_cat = []
            cv_scores_lgbm = []
            oof_pred_cat=np.zeros(X.shape[0])
            oof_pred_lgbm=np.zeros(X.shape[0])


            for idx_train, idx_valid in cv.split(X, prefecture_labels):#   Because it takes a long time to divide the data set, 
                X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]# each time the data set is divided, two models are trained to each other twice, which saves time.
                X_valid, y_valid = X.iloc[idx_valid], y.iloc[idx_valid]
                train_pool=Pool(X_train,y_train,cat_features=self.cat_cols)
                val_pool=Pool(X_valid,y_valid,cat_features=self.cat_cols)
                clf=CatBoostRegressor(**params_cat2,task_type="GPU",devices="0:1",verbose=False)

                clf.fit(train_pool,eval_set=val_pool,verbose=350,early_stopping_rounds=200)
                fitted_models_cat.append(clf)
                y_pred_valid=clf.predict(X_valid)
                oof_pred_cat[idx_valid]=y_pred_valid
                auc_score=mean_absolute_error(y_valid,y_pred_valid)
                cv_scores_cat.append(auc_score)

            
            trial.set_user_attr("models",fitted_models_cat)

            print("CV AUC scores: ", cv_scores_cat)
            print("Maximum CV AUC score: ", max(cv_scores_cat))
            print("Mean CV AUC score: ", mean(cv_scores_cat))
                

            oof_df=pd.DataFrame(index=self.df.index.copy())
            oof_df["pred_cat"]=oof_pred_cat

            return mean(cv_scores_cat)

        sampler = optuna.samplers.TPESampler(multivariate=True)
        study=optuna.create_study(direction="minimize")
        study.optimize(objective,n_trials=20)

        fitted_models_cat=study.best_trial.user_attrs["models"]
        mae_score=study.best_value

        feature_importances_cat=[]

        #-------------------
        #特徴量重要度
        for i in range(len(fitted_models_cat)):
            print(f"Fold-{i}")
            feature_importances = fitted_models_cat[i].get_feature_importance(type='FeatureImportance')

            # 特徴量の重要度をDataFrameに変換
            features = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

            # 特徴量の重要度を降順にソート
            features = features.sort_values(by='Importance', ascending=False)
            feature_importances_cat.append(features)

            # 可視化
            plt.figure(figsize=(8, 20))
            sns.barplot(x='Importance', y='Feature', data=features)
            plt.title('Feature Importance')
            plt.tight_layout()
            plt.show()
        #-------------------
        
        return fitted_models_cat,mae_score

# データの前処理 & 特徴量エンジニアリング
# このクラスは、データの前処理と特徴量エンジニアリングを行います。
# 特に、外部データを使用せずに特徴量を生成することに重点を置いています。
class ProcessData:
    def __init__(self,df_train,df_test):
        self.df_train=df_train
        self.df_test=df_test
        self.df_all=None
        self.cat_cols=None
        self.unique_addresses=None
        self.gdf=None
        self.df_geo=None
        self.process_data1()
        self.process_data2()
        #self.process_data3()
        self.process_data4()
        self.process_data5()
        #self.process_data6()
        self.process_data7()
        self.process_data8()
        #self.jageocoder_data()
        #self.jageocoder_new()
        self.process_data9()
        self.process_data10()
        #self.geopy_test1()
        #self.geopy_test2()
    
    def process_data1(self):
        self.df_all=pd.concat([self.df_train,self.df_test], keys=['train', 'test'])
        print(self.df_all.index)
    
    def process_data2(self):
        nonnull_list=[]
        for col in self.df_all.columns:
            nonnull=self.df_all[col].count()
            if nonnull==0:
                nonnull_list.append(col)
        
        self.df_all=self.df_all.drop(nonnull_list,axis=1)
        self.df_all = self.df_all.drop("市区町村コード", axis=1)
        self.df_all = self.df_all.drop("種類", axis=1)

        on_foot={
            "30分～60分":45,
            "1H～1H30":75,
            "1H30～2H":105,
            "2H～":120
        }
        
        self.df_all["最寄駅：距離（分）"]=self.df_all["最寄駅：距離（分）"].replace(on_foot).astype(float)
        self.df_all["面積（㎡）"]=self.df_all["面積（㎡）"].replace("2,000㎡以上",2000).astype(float)

        #year_now=2023.625
        year_now=2023.99
        y_list={}
        seireki_list={}
        for i in self.df_all["建築年"].value_counts().keys():
            match = re.search(r'\d{4}', i)
            if match:
                num = float(match.group())
                year = year_now - num
                seireki = num
            if "平成" in i:
                num=float(i.split("平成")[1].split("年")[0])
                year=35-num
                seireki=1988+num
            if "令和" in i:
                num=float(i.split("令和")[1].split("年")[0])
                year=5-num
                seireki=2018+num
            if "昭和" in i:
                num=float(i.split("昭和")[1].split("年")[0])
                year=98-num
                seireki=1925+num
            if "戦前" in i:
                num=float(78)
                seireki=float(1945)
            y_list[i]=year
            seireki_list[i]=seireki
        self.df_all["建築西暦年"]=self.df_all["建築年"].replace(seireki_list)
        self.df_all["築年数"]=year_now-self.df_all["建築西暦年"]
        
    def process_data3(self):
        
        self.df_all["取引時点"].value_counts()
        self.df_all["取引年"]=self.df_all["取引時点"].apply(lambda x : int(x[0:4]))
        self.df_all["取引四半期"]=self.df_all["取引時点"].apply(lambda x : int(unicodedata.normalize("NFKC",x[6:7])))
        self.df_all["取引年"]=self.df_all["取引年"]+self.df_all["取引四半期"]*0.25-0.125
    
    def process_data4(self):

        year={
            "年第1四半期":".25",
            "年第2四半期":".50",
            "年第3四半期":".75",
            "年第4四半期":".99"
        }
        year_list={}
        for i in self.df_all["取引時点"].value_counts().keys():
            for k,j in year.items():
                if k in i:
                    year_rep=i.replace(k,j)
            year_list[i]=year_rep
        self.df_all["取引時点"]=self.df_all["取引時点"].replace(year_list).astype(float)


    
    def process_data5(self):

        self.df_all["log_面積（㎡）"]=np.log(self.df_all["面積（㎡）"])
        self.df_all["購入までの築年数"]=self.df_all["取引時点"]-self.df_all["建築西暦年"]
        

    #欠損値の補間
    def process_data6(self):
        #print("process_data3-------------------------------",self.df_all.index)
        #self.df_all[["今後の利用目的","都市計画","改装"]]=self.df_all[["今後の利用目的","都市計画","改装"]].fillna("unknown")
        #####self.df_all=self.df_all.drop("取引四半期",axis=1)
        pass

    def process_data7(self):
        self.df_all["面積（㎡）-最寄駅：距離（分）"]=self.df_all["面積（㎡）"]-self.df_all["最寄駅：距離（分）"]
        self.df_all["面積（㎡）-築年数"]=self.df_all["面積（㎡）"]-self.df_all["築年数"]
        self.df_all["面積（㎡）容積率（％）_combi"] = self.df_all["面積（㎡）"] * self.df_all["容積率（％）"]
        """
        durations = []
        for i in range(len(self.df)):
            if self.df["取引年"].iloc[i]-self.df["建築西暦年"].iloc[i] >= 0:
                duration = self.df["取引年"].iloc[i] - self.df["建築西暦年"].iloc[i]
            else:
                duration=numpy.nan
            durations.append(duration)
       
        self.df["購入までの築年数"] = durations
        """
        self.df_all["旧耐震フラグ"]=0
        self.df_all["旧耐震フラグ"].mask(self.df_all["建築西暦年"]<=1981,1,inplace=True)
        
    
    def process_data08(self):
        if "取引価格（総額）_log" in self.df_all.columns:
            self.df_all["取引価格（総額）_log"]=np.log10(10**self.df_all["取引価格（総額）_log"]/self.df_all["面積（㎡）"])
            self.df_all.rename(columns={"取引価格（総額）_log":"単位面積あたりの取引価格_log"},inplace=True)
        else:
            print("カラム ⇒ 取引価格（総額）_log が存在しなかったので 単位面積あたりの取引価格_log は計算しませんでした")
    
    def process_data8(self):
        if "取引価格（総額）_log" in self.df_all.columns:
            self.df_all["単位面積あたりの取引価格_log"]=np.log10(10**self.df_all["取引価格（総額）_log"]/self.df_all["面積（㎡）"])
            #self.df_all.rename(columns={"取引価格（総額）_log":"単位面積あたりの取引価格_log"},inplace=True)
        else:
            print("カラム ⇒ 取引価格（総額）_log が存在しなかったので 単位面積あたりの取引価格_log は計算しませんでした")
    
    def jageocoder_new(self):
        geo=Geocod2(self.df_all)
        geo.make_unique_addresses()

    
    def jageocoder_data(self):
        geo=Geocod(self.df_all)
        geo.make_unique_addresses()
        df_geo,ad_=geo.search_addresses()
        self.df_geo=df_geo
        df_geo=df_geo[["address","matched","x","y","level","priority"]]

        self.df_all["address"]=self.df_all["都道府県名"].apply(lambda x:"" if pd.isna(x) else x)+self.df_all["市区町村名"].apply(lambda x:"" if pd.isna(x) else x)+self.df_all["地区名"].apply(lambda x:"" if pd.isna(x) else x)
        self.df_all = self.df_all.join(df_geo.set_index('address'), on='address', how='left')

        self.df_all=self.df_all.drop("matched",axis=1)
        self.df_all=self.df_all.drop([
                                    #"住所",
                                    "address"
                                    ],axis=1)
    
        

    def process_data9(self):
        self.df_all,self.cat_cols=to_cat_col(self.df_all)

    
    def process_data10(self):
        # 'train' データを取得
        self.df_train = self.df_all.loc[('train',)]

        # 'test' データを取得
        self.df_test = self.df_all.loc[('test',)]
    
    def filter_data(self):
        key="address"
        d=self.df_train[f"{key}"].unique()
        print(len(d))
        t=self.df_test[f"{key}"].unique()
        print(len(t))

        # Bのデータフレームに存在しているがAのデータフレームには存在しないデータを取得
        missing_data = self.df_test[~self.df_test[f"{key}"].isin(self.df_train[f"{key}"])]
        #print(missing_data)

        # Bのデータフレームに存在するIDのデータだけAのデータも取得
        result_df= self.df_train[self.df_train[f"{key}"].isin(self.df_test[f"{key}"])]
        #result_df = self.df_train[common_ids]

    def return_data(self):
        return self.df_train,self.df_test,self.df_all
    
    def return_data_test(self):
        return self.df_train,self.df_test,self.df_all,self.df_geo,self.cat_cols


class Geocod2:
    def __init__(self, df):
        self.df = df
        self.df_result = pd.DataFrame(columns=['matched', 'address', 'id', 'name', 'x', 'y', 'level', 'priority', 'note', 'fullname'])
        self.df_search_results = None

    def make_unique_addresses(self):
        self.df["住所"] = self.df.apply(lambda row: f"{row['都道府県名']}{row['市区町村名']}{row['地区名']}", axis=1)
        self.unique_addresses = self.df["住所"].drop_duplicates().tolist()
        print("ユニークなアドレスの数=>", len(self.unique_addresses))
        print(self.unique_addresses)

    def search_addresses(self):
        address_list = self.unique_addresses
        all_results = []

        for i, address in enumerate(tqdm_notebook(address_list)):
            jageocoder.init()
            result = jageocoder.search(address)
            address_df = pd.DataFrame([address], columns=["address"])
            matched_df = pd.DataFrame([result['matched']], columns=['matched'])
            matched_df = pd.concat([matched_df, address_df], axis=1)
            candidates_df = pd.DataFrame(result['candidates'])
            #print("外側のfor文で取得")
            #print(candidates_df)
            #print("---------------------------------------------------")

            for _, candidate in candidates_df.iterrows():
                #print("内側のfor文で取得")
                #print(candidate)
                #print("--------------------------------------------------------------")
                temp_df = pd.concat([matched_df.reset_index(drop=True), candidate.to_frame().T.reset_index(drop=True)], axis=1)
                #print("match_dfと結合")
                #print(temp_df)
                #print("--------------------------------------------------------------")
                self.df_result = pd.concat([self.df_result, temp_df], ignore_index=True)
                #print("self.df_resultと結合")
                #print(self.df_result)
                #print("--------------------------------------------------------------")
            #print("--------------------------------------------------------------")

        #self.df_result = pd.concat([self.df_result, pd.Series(address_list, name="住所").reset_index(drop=True)], axis=1)
        self.df_search_results = self.df_result.dropna(subset=["matched"])
        print(len(self.df_search_results))
        self.df_search_results.to_csv(r"C:\Users\yuto2\OneDrive\ドキュメント\Pythonフォルダ\Nishika\中古マンション価格予測_2024夏の部\作成データ\df_search_results.csv")

        return self.df_search_results, address_list





class Geocod:
    def __init__(self,df):
        self.df=df
        self.df_result=pd.DataFrame()
        self.df_search_results=None
    
    def make_unique_addresses(self):
        self.df["住所"]=self.df["都道府県名"].apply(lambda x:"" if pd.isna(x) else x)+self.df["市区町村名"].apply(lambda x:"" if pd.isna(x) else x)+self.df["地区名"].apply(lambda x:"" if pd.isna(x) else x)
        self.unique_addresses=self.df.copy()

        #ユニークな組合せを抽出
        self.unique_addresses["住所"]=self.unique_addresses["都道府県名"].apply(lambda x:"" if pd.isna(x) else x)+self.unique_addresses["市区町村名"].apply(lambda x:"" if pd.isna(x) else x)+self.unique_addresses["地区名"].apply(lambda x:"" if pd.isna(x) else x)
        print(self.unique_addresses)
        self.unique_addresses=self.unique_addresses[["住所"]].drop_duplicates()

        print("ユニークなアドレスの数=>",len(self.unique_addresses))
        print(self.unique_addresses)
    
    def make_unique_addresses1(self):
        self.df["住所"] = self.df.apply(lambda row: f"{row['都道府県名']}{row['市区町村名']}{row['地区名']}", axis=1)
        self.unique_addresses = self.df["住所"].drop_duplicates().tolist()
        print("ユニークなアドレスの数=>", len(self.unique_addresses))
        print(self.unique_addresses)

    def search_addresses(self):
        #address_list=self.unique_addresses["住所"]
        address_list=self.unique_addresses
        #print(address_list)
        #print(len(address_list))
        # 全ての検索結果を格納するリスト
        all_results = []

        # 各住所に対して検索を実行し、結果をデータフレームに追加
        for i,address in enumerate(tqdm_notebook(address_list)):
            jageocoder.init(
                #db_dir="/usr/jageocoder/db2"
                )
            result = jageocoder.search(address)
            #print("ループ=>",i)
            #print("入力されたアドレス=>",address)
            #print(result)
            address_df=pd.DataFrame([address],columns=["address"])

            matched_df = pd.DataFrame([result['matched']], columns=['matched'])
            matched_df=pd.concat([matched_df,address_df],axis=1)
            #print(matched_df)
            candidates_df = pd.DataFrame(result['candidates'])
            #print(candidates_df)

            temp_df = pd.concat([matched_df, candidates_df],axis=1)
            self.df_result = pd.concat([self.df_result,temp_df], ignore_index=True)

             # ジオコード前の住所をself.df_search_resultsに追加
            #self.df_result = pd.concat([self.df_result, pd.DataFrame([address], columns=['address'])], ignore_index=True,axis=1)
            #print(self.df_result)
        #df_address=pd.DataFrame(address_list,columns=["address"])
        #self.df_result=pd.concat([self.df_result,address_list.reset_index(drop=True)],axis=1)
        
        self.df_search_results=self.df_result.dropna(subset=["matched"])
        print(len(self.df_search_results))
        self.df_search_results.to_csv(r"C:\Users\yuto2\OneDrive\ドキュメント\Pythonフォルダ\Nishika\中古マンション価格予測_2024夏の部\作成データ\df_search_results.csv")

        return self.df_search_results,address_list
    
    #元のデータフレームの住所カラムに対して全てジオコーディングする。
    def search_addresses_by_force(self):
        # 各カラムの値がNaNの場合、空文字として結合
        self.df["住所"] = self.df["都道府県名"].fillna("") + self.df["市区町村名"].fillna("") + self.df["地区名"].fillna("")
        address_list=self.df["住所"]
        print("address_list個数",len(address_list))

        # 各住所に対して検索を実行し、結果をデータフレームに追加
        for i,address in enumerate(tqdm_notebook(address_list)):
            jageocoder.init()
            result = jageocoder.search(address)
            print("ループ=>",i)
            print("入力されたアドレス=>",address)
            print(result)
            matched_df = pd.DataFrame([result['matched']], columns=['matched'])
            candidates_df = pd.DataFrame(result['candidates'])
            temp_df = pd.concat([matched_df, candidates_df],axis=1)
            self.df_result = pd.concat([self.df_result, temp_df], ignore_index=True)

        print("self.df",len(self.df))
        print("self.df_result",len(self.df_result))
        self.df=pd.concat([self.df,self.df_result],axis=1)

        return self.df

class Geocod_API:
    def __init__(self,df):
        self.df=df
        self.df_result=pd.DataFrame()
        self.df_search_results=None
    
    def make_unique_addresses(self):
        self.df["住所"]=self.df["都道府県名"].apply(lambda x:"" if pd.isna(x) else x)+self.df["市区町村名"].apply(lambda x:"" if pd.isna(x) else x)+self.df["地区名"].apply(lambda x:"" if pd.isna(x) else x)
        self.unique_addresses=self.df.copy()

        #ユニークな組合せを抽出
        self.unique_addresses["住所"]=self.unique_addresses["都道府県名"].apply(lambda x:"" if pd.isna(x) else x)+self.unique_addresses["市区町村名"].apply(lambda x:"" if pd.isna(x) else x)+self.unique_addresses["地区名"].apply(lambda x:"" if pd.isna(x) else x)
        #print(self.unique_addresses)
        self.unique_addresses=self.unique_addresses.drop_duplicates(subset=["住所"])

        print("ユニークなアドレスの数=>",len(self.unique_addresses))
        #print(self.unique_addresses)
    
    def search_addresses3(self):
        #self.unique_addresses=self.unique_addresses.reset_index()
        #print(self.unique_addresses.columns)
        print("ユニークなアドレスの数=>",len(self.unique_addresses))
        #print(address_list)
        #print(len(address_list))
        # 全ての検索結果を格納するリスト
        all_results = []
        # データを格納するリスト
        data_list = []

        # 各住所に対して検索を実行し、結果をデータフレームに追加
        for i,(index,row) in enumerate(tqdm_notebook(self.unique_addresses.iterrows(),total=len(self.unique_addresses))):
            address=row["住所"]
            # 住所をURLエンコード
            encoded_address = urllib.parse.quote(address)
            # APIのURL
            url = f"https://msearch.gsi.go.jp/address-search/AddressSearch?q={encoded_address}"
            # APIリクエストを送信
            response = requests.get(url)
            # レスポンスをJSON形式で取得
            results = response.json()
            #print("ループ=>",i)
            #print("入力されたアドレス=>",address)
            #print("国土地理院APIからの出力=>",results)
            #print("row",row)

            #全ての候補を保存
            if results:
                for result in results:
                    title=result['properties']['title']
                    #print(title)
                    coordinates = result['geometry']['coordinates']
                    #print(coordinates)
                    all_results.append({
                        '都道府県名':row["都道府県名"],
                        '市区町村名':row["市区町村名"],
                        '地区名':row["地区名"],
                        '最寄駅：名称':row["最寄駅：名称"],
                        '最寄駅：距離（分）':row["最寄駅：距離（分）"],
                        'address':address, 
                        'title': title, 
                        'x': coordinates[0], 
                        'y': coordinates[1]})
                    #print(all_results)
            else:
                # 結果が空の場合でもデータを追加
                all_results.append({
                        '都道府県名':row["都道府県名"],
                        '市区町村名':row["市区町村名"],
                        '地区名':row["地区名"],
                        '最寄駅：名称':row["最寄駅：名称"],
                        '最寄駅：距離（分）':row["最寄駅：距離（分）"],
                        'address': address, 
                        'title': None, 
                        'x': None, 
                        'y': None})
            
            # 各住所についてデータを取得
            if results:
                # 最初の結果のみを使用
                result = results[0]
                title = result['properties']['title']
                coordinates = result['geometry']['coordinates']
                data_list.append({
                        '都道府県名':row["都道府県名"],
                        '市区町村名':row["市区町村名"],
                        '地区名':row["地区名"],
                        '最寄駅：名称':row["最寄駅：名称"],
                        '最寄駅：距離（分）':row["最寄駅：距離（分）"],
                        'address':address, 
                        'title': title, 
                        'x': coordinates[0], 
                        'y': coordinates[1]})
                #print(data_list)
            else:
                # 結果が空の場合でもデータを追加
                data_list.append({
                        '都道府県名':row["都道府県名"],
                        '市区町村名':row["市区町村名"],
                        '地区名':row["地区名"],
                        '最寄駅：名称':row["最寄駅：名称"],
                        '最寄駅：距離（分）':row["最寄駅：距離（分）"],
                        'address': address, 
                        'title': None, 
                        'x': None, 
                        'y': None})
            
            #if i==100:
            #    break
    
        # データフレームに変換
        df_all_address = pd.DataFrame(all_results)
        print("全てのデータのデータフレーム")
        print(df_all_address)
        df_address=pd.DataFrame(data_list)
        print("データフレーム")
        print(df_address)

        return df_all_address,df_address


def optimize_SARIMA1(endog:Union[pd.Series,list],order_list,d,D,s):#Union[pandas.Series,list]はpandas.Seriseかリストと引数として受取る事を表す。
    #次数(p,q)とそのAICをタプルとして格納するために空のリストを初期化
    results=[]

    #(p,q)の一意な組み合わせをループ処理
    for order in tqdm_notebook(order_list):
        #print(order)
        try:
            #SARIMAX関数を使ってARIMA(p,d,q)モデルを適合させる
            model=SARIMAX(endog,
                          order=(order[0],d,order[1]),
                          seasonal_order=(order[2],D,order[3],s),
                          simple_differencing=False).fit(disp=False)#order=(自己回帰(AR)項,統合(I)項,移動平均(MA)項)
        except:
            continue

        #モデルのAICを計算
        aic=model.aic
        #(p,q)の組み合わせとAICをタプルとして結果リストに追加
        results.append([order,aic])
    
    #(p,q)の組み合わせとAICをDataFrameに格納
    result_df=pd.DataFrame(results)
    #DataFrameの列にラベルを付ける
    result_df.columns=["(p,q,P,Q)","AIC"]

    #昇順でソート:AICの値が小さいほど、よいモデルである。
    result_df=result_df.sort_values(by="AIC",ascending=True).reset_index(drop=True)

    return result_df

def optimize_SARIMA(endog, order_list, d_list, D_list, s):
    results = []

    for order in tqdm_notebook(order_list):
        for d in d_list:
            for D in D_list:
                try:
                    model = SARIMAX(endog,
                                    order=(order[0], d[0], order[1]),
                                    seasonal_order=(order[2], D[0], order[3], s),
                                    simple_differencing=False).fit(disp=False)
                    aic = model.aic
                    results.append([(order[0], d[0], order[1], order[2], D[0], order[3]), aic])
                except Exception as e:
                    print(f"Error fitting model: {e}")
                    continue

    result_df = pd.DataFrame(results, columns=["(p,d,q,P,D,Q)", "AIC"])
    result_df = result_df.sort_values(by="AIC", ascending=True).reset_index(drop=True)

    # 最小のAICを持つパラメータの組み合わせを取得
    best_params = result_df.iloc[0]["(p,d,q,P,D,Q)"]
    best_aic = result_df.iloc[0]["AIC"]

    return result_df,best_params, best_aic


def check_dataframe_elements(df):
    df.info()
    print("マルチインデックス=>",isinstance(df.index,pd.MultiIndex))
    print(df.shape)
    print("-----カラム名一覧-----")
    print(df.columns)
    for col in df.columns.to_list():
        print(f"-----{col}-----に含まれるユニークな値の数")
        print(df[col].nunique())
        print(f"-----{col}-----に含まれる値を被り無しで表示")
        print(df[col].unique())

def submisson_datetime():
    now=datetime.now()
    now_str=now.strftime("%Y%m%d_%H%M%S")

    return now_str

def folium_test(df,boundary_geojson=None):
    #地図の基本設定
    m=folium.Map(location=[40.7128, -74.0060], zoom_start=11)

    # カウンターを初期化
    count = 0
    max_points = 5

    for index,row in df.iterrows():
        if pd.notnull(row["y"]) and pd.notnull(row["x"]):
            #if row["health"]==1:
            #    marker_color="green"
            #elif row['health'] == 2:
            #    marker_color = 'red'
            #else:
            #    marker_color = 'blue'
            

            # マーカーを追加
            folium.CircleMarker(
            location=(row['y'], row['x']),
            radius=5,
            #color=marker_color,
            fill=True,
            fill_opacity=0.7
            ).add_to(m)
        
            # カウンターを増加
            count += 1
            # 5つの点を表示したらループを終了
            if count >= max_points:
                break

    # 地図を保存    
    m.save(r"C:\Users\yuto2\OneDrive\ドキュメント\Pythonフォルダ\Nishika\中古マンション価格予測_2024春の部\作成データ\map.html")


# 表記ゆれを修正
def trans_chara(text):
    word_from = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ一二三四五六七八九"
    word_to = "０１２３４５６７８９ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ１２３４５６７８９"
    translation_table = str.maketrans(word_from, word_to)
    return text.translate(translation_table)

# 欠損値を補完する関数
def fill_missing_values(df, key_columns, target_column):
    for keys, group in df.groupby(key_columns):
        #print("keys",keys,"group",group)
        most_common_value = group[target_column].mode().iloc[0] if not group[target_column].mode().empty else np.nan
        df.loc[group.index, target_column] = df.loc[group.index, target_column].fillna(most_common_value)
    return df


# 最寄り駅の欠損値を補完
def fill_missing_with_mode(df, target_col, group_col):
    # グループごとの最頻値を計算
    mode_values = df.groupby(group_col)[target_col].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
    #check_dataframe_elements(mode_values)

    # 欠損値を補完する関数
    def fill_value(row):
        #target_col列が欠損値か確認
        if pd.isnull(row[target_col]) or row[target_col]=="unknown":
            if row[group_col] in mode_values.index:
                print("欠損値補間発動")
                print(f"{row[group_col]} -> {mode_values[row[group_col]]}")
                return mode_values[row[group_col]]
            else:
                #欠損値であり、最頻値も無い場所は欠損値を返す
                print("欠損値ありで、最頻値もない")
                print(row[target_col])
                return row[target_col]  # 欠損値のままにする
        else:
            #欠損値でない場所はそのままの値を返す
            #print(row[target_col])
            return row[target_col]

    # target_colの欠損値を補完
    df[target_col] = df.apply(fill_value, axis=1)

    return df

def add_longitude_latitude(df,df_station,df_address):
    # IDを退避
    #df = df.reset_index()
    # 地区名はNanが存在するため、fillnaする
    #df['住所'] = df["都道府県名"].apply(lambda x:"" if pd.isna(x) else x)+df["市区町村名"].apply(lambda x:"" if pd.isna(x) else x)+df["地区名"].apply(lambda x:"" if pd.isna(x) else x)
    df['住所'] = df['都道府県名'] + df['市区町村名'] + df['地区名'].fillna('')
    #　住所と最寄駅：名称の最頻値で欠損値を補完
    df = fill_missing_with_mode(df, '最寄駅：名称', '住所')
    aaa=df.copy()
    # 外部データに名前寄せする
    df['最寄駅：名称'] = (
        df['最寄駅：名称']
        .fillna('unknown')
        .str.replace(r'\(.*\)', '', regex=True)
        .str.replace('ケ', 'ヶ')
        .str.replace('\u3000', ' ') # トヨタモビリティ富山 Gスクエア五福前に含まれている
        .str.replace('祗園橋', '祇園橋')
        .apply(trans_chara)
    )
    df = pd.merge(df, df_station, on=['最寄駅：名称'], how='left')
    
    #print(df)
    # mask = 最寄り駅に基づいて緯度・経度が算出できなかった行
    mask = df['lon'].isnull()
    # 住所から緯度・経度を算出
    #print(df.loc[mask, '住所'])
    print()

     # df_address とマージして緯度・経度を一括で取得
    df_address = df_address.rename(columns={'address': '住所'})
    df = pd.merge(df, df_address[['住所', 'x_api', 'y_api']], on='住所', how='left')
    #print(df.loc[mask, 'lon'],df.loc[mask, 'x_api'])
    df.loc[mask, 'lon'] = df.loc[mask, 'x_api']
    
    df.loc[mask, 'lat'] = df.loc[mask, 'y_api']
    #geocoded_data = df.loc[mask, '住所'].apply(lambda x: pd.Series(get_lat_lon(x,df_address), index=['lon', 'lat']))
    #df.update(geocoded_data)
    #df = df.set_index('ID')
    #df = df.drop(['住所'], axis=1)

    return df,mask,aaa


# 漢数字を全角数値に変換する関数
def convert_kanji_to_fullwidth_2(text):
    kanji_to_fullwidth = {
        '一': '１', '二': '２', '三': '３', '四': '４', '五': '５',
        '六': '６', '七': '７', '八': '８', '九': '９', '〇': '０'
    }
    for kanji, fullwidth in kanji_to_fullwidth.items():
        text = text.replace(kanji, fullwidth)
    
    # 十の処理を追加
    text = re.sub(r'([１-９０])十([１-９０])', r'\1\2', text)  # 両隣が全角数値の場合
    text = re.sub(r'十([１-９０])', r'１\1', text)  # 右隣が全角数値の場合
    text = re.sub(r'([１-９０])十', r'\1０', text)  # 左隣が全角数値の場合
    text = re.sub(r'十', '１０', text)  # 隣が全角数値でない場合

    # 大字や字の処理を追加
    text = text.replace('大字', '')
    text = text.replace('字', '')

    # 糟を粕に変換する処理を追加
    text = text.replace('糟', '粕')
    
    return text

# 欠損値を補完する関数
def find_nearest_station_main_table(row, df):
    # 経度や緯度が欠損値の場合は欠損値を返す
    if pd.isna(row['x_api']) or pd.isna(row['y_api']):
        return np.nan
    
    if row['最寄駅：名称'] != "ｕｎｋｎｏｗｎ":
        return row['最寄駅：名称']
    
    min_distance = float('inf')
    nearest_station = None
    
    for _, other_row in df.iterrows():
        # 自分自身の行はスキップ
        if row.name == other_row.name:
            continue
        
        # other_rowの経度や緯度が欠損値の場合はスキップ
        if pd.isna(other_row['lat']) or pd.isna(other_row['lon']):
            continue
        
        distance = geodesic((row['y_api'], row['x_api']), (other_row['lat'], other_row['lon'])).meters
        if distance < min_distance:
            min_distance = distance
            nearest_station = other_row['最寄駅：名称']
    
    return nearest_station


# 距離を計算する関数
def calculate_distance(row):
    if pd.isna(row['y_api']) or pd.isna(row['x_api']) or pd.isna(row['lat']) or pd.isna(row['lon']):
        return None
    address_coords = (row['y_api'], row['x_api'])
    station_coords = (row['lat'], row['lon'])
    return geodesic(address_coords, station_coords).meters


# Geohashを取得する関数
def encode_geohash(row, precision):
    if pd.isna(row['y_api']) or pd.isna(row['x_api']):
        return None
    return geohash.encode(row['y_api'], row['x_api'], precision=precision)


def mae_loss_fn(true_targets,pred_weighted):
    return mean_absolute_error(true_targets,pred_weighted)

class WeightsSearcher:
    def __init__(self,loss_fn,bounds=[],mode="min",method="SLSQP"):
        self.loss_fn=loss_fn#損失関数
        self.bounds=bounds#重みの範囲
        self.mode=mode#最適化モード(minまたはmax)
        self.method=method#最適化方法(デフォルトは"SLSQP")
    
    # 目的関数のラッパー: 予測値、真のターゲット、目的関数を受け取り、重み付けされた予測値を計算する関数を返す
    def _objective_function_wrapper(self,pred_values,true_targets,obj_fn):
        def objective_function(weights):
            #print(pred_values)
            #予測値に重みを適用し、行ごとに合計
            pred_weighted=(pred_values*weights).sum(axis=1)
            #真のターゲットと重み付けされた予測値を使用してスコアを計算
            score = obj_fn(true_targets, pred_weighted)
            # 最適化モードに応じてスコアを反転
            return score if self.mode == "min" else score
        #目的関数を返す
        return objective_function
    
    #最適な重みを見つけるメソッド:検証予測値と真のターゲットを受け取り、最適化を実行
    def find_weights(self, val_preds, true_targets):
        len_models = len(self.bounds)#モデルの数を取得
        bounds = [0,1] * len_models if len(self.bounds) == 0 else self.bounds# 重みの範囲が指定されていない場合は [0,1] の範囲を設定
        ##np.ones は、指定された形状とデータ型で、すべての要素が1である新しい配列を生成するNumPy関数です。
        ##len_models はモデルの数を表しており、この数だけ1が含まれた配列が作成されます。
        ##例えば、len_models が3の場合、np.ones(len_models) は [1, 1, 1] という配列を生成します。
        ###initial_weights = np.ones(len_models) / len_models# 初期重みを設定（各モデルに等しい重み）
        initial_weights = np.random.rand(len_models)  # 初期重みをランダムに設定
        initial_weights /= np.sum(initial_weights)  # 初期重みを正規化
        objective_function = self._objective_function_wrapper(val_preds, true_targets, self.loss_fn)#目的変数をラップ

        # 制約条件: 重みの合計が1になるようにする
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

        result = minimize(
            objective_function, 
            initial_weights, 
            bounds=bounds, 
            method=self.method,
            constraints=constraints
        )
        #最適化された重みを取得
        optimized_weights = result.x
        #重みの合計が1になるように正規化
        optimized_weights /= np.sum(optimized_weights)
        #最適化された重みを返す
        return optimized_weights