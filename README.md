# Projeto VIA
Desafio da VIA para vaga de Sênior Especialista - Machine Learning

## Contexto
Realizamos nesse projeto a construção de um modelo de Machine Learning que classifique as flores em suas respectivas espécies.

## Análise do Modelo
O problema foi abordado fazendo uma análise do modelo realizando um treino, teste com o dataset e analisando as métricas e a importância das features do modelo. Toda a implementação foi realizada utilizando Python.

conf. Projeto IRIS - VIA.ipynb

## Endpoint
A API foi disponibilizada para consultar o modelo online no endpoint com o método REST 'POST'.
```
http://35.199.77.88:5000/predict/
```
Como parâmetro, esse endpoint aceita um JSON (application/JSON) com construído da seguinte forma:
```
{
  "feature_0": "10",
  "feature_1": "10",
  "feature_2": "10",
  "feature_3": "10"
}
```
O resultado da API também é devolvido como JSON e será parecido com o formato seguinte:
```
{
"data": {
"classe": "Iris-virginica",
"feature 0": 10,
"feature 1": 10,
"feature 2": 10,
"feature 3": 10,
"resultado": 3
}
}
```

## Interface Web
Uma interface web foi desenvolvida para interagir com o modelo. Nessa interface é possível colocar os valores numéricos nos campos de features e clicar no butão **validar** para obter a classificação do modelo.

````
http://35.199.77.88:8501/
````