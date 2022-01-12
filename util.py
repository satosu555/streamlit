#util.py
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
# 回帰結果評価用関数
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
import streamlit as st
import numpy as np

# draw confusion matrix
def draw_confusion_matrix(confusion_matrix, class_names):
    fig = plt.figure(figsize=(6, 6))
    heatmap = sns.heatmap(
        confusion_matrix, xticklabels=class_names, yticklabels=class_names,
        annot=True, fmt="d", cbar=True, square=True, cmap='YlGnBu')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    return fig

def draw_confusion_matrix_plotly(confusion_matrix, class_names):
    fig = go.Figure(data=go.Heatmap(
        z=confusion_matrix,
        x=class_names,
        y=class_names,
        hoverongaps = False,
        colorscale="tempo"))
    fig.update_layout(title={"text":"confusion_matrix"})
    fig.update_xaxes(title={"text":"Predict label"})
    fig.update_yaxes(title={"text":"True label"})
    return fig

def Regression_evaluator(y_test, y_pred):
    # 正答率
    st.write("正答率:", round(accuracy_score(y_test, y_pred), 3) * 100)
    # 平均絶対誤差 (MAE: Mean Absolute Error)
    st.write("平均絶対誤差:", round(mean_absolute_error(y_test, y_pred),3))
    # 平均二乗誤差 (MSE: Mean Squared Error)
    st.write("平均絶対誤差:", round(mean_squared_error(y_test, y_pred),3))
    # 二乗平均平方根誤差 (RMSE: Root Mean Squared Error) -- sklearn には実装されていないのでnumpyを利用
    st.write("二乗平均平方根誤差:", round(np.sqrt(mean_squared_error(y_test, y_pred)),3))
    # 相関係数（Correlation)
    st.write("相関係数 (R):", round(np.sqrt(r2_score(y_test, y_pred)),3))
    # 決定係数 (R2, R-squared, coefficient of determination)) 
    st.write("決定係数 (R2):", round(r2_score(y_test, y_pred),3))
    return