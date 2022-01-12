# modified 2022/01/10 by Atsuhiko Ii
# updated 2022/01/12 by Akihiro Suto
import os
from datetime import datetime
import pandas as pd
from util import * # 自作モジュール
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import streamlit as st
import pandas as pd

st.sidebar.title(" # CIDS AutoML \n Streamlit ver.")
st.sidebar.markdown("https://streamlit.io/")
st.sidebar.markdown("## Please select dataset and algorithm")

# データとモデルの選択 （サイドバーにて）
data_set = st.sidebar.radio("Dataset:",('Wine', 'Iris'))
model_name = st.sidebar.radio("Algorithm:",("Decision Tree","Support Vector Machine"))

# st.write("Dataset:", data_set) 
# st.write("Algorithm:", model)

if data_set == "Iris":
    from sklearn.datasets import load_iris
    data = load_iris()
elif data_set == "Wine":
    from sklearn.datasets import load_wine
    data = load_wine()
else:
    st.write("Please select iris or wine.")

if model_name == "Decision Tree":
    model = tree.DecisionTreeClassifier()
elif model_name == "Support Vector Machine":
    model = SVC(kernel='linear', probability=True)
else:
    st.write("Please select dt or svc.")

df_X = pd.DataFrame(data.data, columns=data.feature_names)
df_y = pd.DataFrame(data.target, columns=["target_names"])

# 標準化をいれるならここ

# 訓練用と検証用のデータセット作成
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.3, random_state=0)

# 予測モデルを作成
model.fit(X_train, y_train)
pred = model.predict(X_test)
y_pred = pd.DataFrame(pred)
y_pred = model.predict(X_test)

# モデル評価用の関数
Regression_evaluator(y_test,y_pred)

# グラフの描画
# Plotlyはこっち → st.plotly_chart(figure_or_data, use_container_width=False, sharing="streamlit", **kwargs)
# Matplotlibはこっち → st.pyplot(fig=None, clear_figure=None, **kwargs)

# fig = draw_confusion_matrix(confusion_matrix(y_test, y_pred), data.target_names)
fig_plotly = draw_confusion_matrix_plotly(confusion_matrix(y_test, y_pred), data.target_names)

# st.pyplot(fig)
st.plotly_chart(fig_plotly)

# 元データの表示
st.header("Original Data")
st.write(data)