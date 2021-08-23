from __future__ import print_function, absolute_import

from pyspark import SparkContext, SparkConf, SparkFiles
from pyspark.sql import SparkSession

from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler, RobustScaler
from pyspark.ml import Pipeline

from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import numpy as np
import os
import joblib
import streamlit as st 

# Contexto Spark
sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

# URL de dados
url_dataset = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# Diretório do modelo
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

# Dicionário
dictionary = {'Iris-setosa':1, 'Iris-versicolor':2, 'Iris-virginica':3}

class ViaModel:
    def __init__(self):
        self.df = None

    def getData(self):
        """
        Leitura dos dados.
        """
        self.df = pd.read_csv(url_dataset, header=None)
        return
    
    def transformData(self):
        """
        Transformação do target.
        """
        self.df['label'] = self.df[4].map(dictionary)
        self.df_spark = spark.createDataFrame(self.df)

        # Criando as features
        vectorAssembler = VectorAssembler(inputCols=['0', '1', '2', '3'], outputCol='features')
        self.v_df = vectorAssembler.transform(self.df_spark)
        self.v_data_df = self.v_df.select(['features', 'label'])
        return

    def splitData(self):
        # Dataset de Treino de Teste.
        splits = self.v_data_df.randomSplit([0.66, 0.34])
        xTrain_dataset = splits[0]
        xTest_dataset = splits[1]
        return xTrain_dataset, xTest_dataset

    def buildModel(self, x, y):
        """
        Construção do Modelo de Classificação.
        """
        nsamples, nx, ny = np.shape(x)
        self.rd_forest = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
        self.rd_forest.fit(np.array(x).reshape(nsamples, nx*ny), np.array(y).ravel())
        # Salvando o modelo.
        joblib.dump(self.rd_forest, os.path.join(CURRENT_DIR, 'iris_model.pkl'))
        return
    
    def loadModel(self):
        self.rd_forest = joblib.load(os.path.join(CURRENT_DIR, 'iris_model.pkl'))
        return

    def setData(self, feature_0, feature_1, feature_2, feature_3):
        """
        Features para teste do modelo de ML.
        """
        self.feature_0 = feature_0
        self.feature_1 = feature_1
        self.feature_2 = feature_2
        self.feature_3 = feature_3
        return 
    
    def predict(self):
        x_test = np.array([self.feature_0, self.feature_1, self.feature_2, self.feature_3])
        predict = self.rd_forest.predict(np.array(x_test).reshape(1, -1))
        return predict

    def train(self):
        """
        Treinamento do modelo.
        """
        self.getData()
        self.transformData()
        xTrain_dataset, xTest_dataset = self.splitData()
        x = xTrain_dataset.select('features').collect()
        y = xTrain_dataset.select('label').collect()
        self.buildModel(x, y)

via = ViaModel()
if not os.path.join(CURRENT_DIR, 'iris_model.pkl'):
    via.train()
else:
    via.loadModel()

st.markdown("<h4 style='text-align: center;'>PROJETO VIA - MACHINE LEARNING</h4>", unsafe_allow_html=True)

with st.form(key='model_form'):
    feat_0, feat_1, feat_2, feat_3 = st.beta_columns(4)

    with feat_0:
        f0 = st.number_input("Feature 0")
    
    with feat_1:
        f1 = st.number_input("Feature 1")

    with feat_2:
        f2 = st.number_input("Feature 2")
    
    with feat_3:
        f3 = st.number_input("Feature 3")
    
    submit_1 = st.form_submit_button('Validar')

if submit_1:
    via.setData(f0, f1, f2, f3)
    p = via.predict()
    for k, v in dictionary.items():
        if v == p[0]:
            st.markdown("<h4 style='text-align: center;'>Resultado: {}</h4>".format(k), unsafe_allow_html=True)