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
from flask import Flask, jsonify, request


# Context Flask
app = Flask(__name__)

# Contexto Spark
conf = SparkConf().setAppName("Application").set("spark.executor.heartbeatInterval", "200000").set("spark.network.timeout", "300000")
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
if os.path.isfile(os.path.join(CURRENT_DIR, 'iris_model.pkl')):
    via.loadModel()
else:
    via.train()

@app.route('/', methods=["GET", "POST"])
def start():
    return jsonify({'projeto': 'Desafio via', 'autor': 'Charles Akono'})

@app.route('/predict/', methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        f0 = np.float(request.json['feature_0'])
        f1 = np.float(request.json['feature_1'])
        f2 = np.float(request.json['feature_2'])
        f3 = np.float(request.json['feature_3'])
        via.setData(f0, f1, f2, f3)
        p = via.predict()
        predict_classe = None
        for k, v in dictionary.items():
             if v == p[0]:
                 predict_classe = k
    return jsonify(
        {
            "data": {
                "feature 0": f0,
                "feature 1": f1,
                "feature 2": f2,
                "feature 3": f3,
                "resultado": int(p[0]),
                "classe": str(predict_classe),
            }
        }
    )

if __name__ == "__main__":
    app.run()
