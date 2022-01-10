#app.py
# modified 2022/01/10 by Atsuhiko Ii
import os
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, g, flash
import pandas as pd
from util import draw_confusion_matrix # 自作モジュール
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

app = Flask(__name__)

SAVE_DIR = "graph"
if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)

@app.route('/graph/<path:filepath>')
def register_folder(filepath):
    return send_from_directory(SAVE_DIR, filepath)  

@app.route("/", methods=["GET","POST"])
def main():
    if request.method == "GET":
        return render_template("index.html")

    if request.method == "POST":

        data_name = request.form.get('data_name')
        model_name = request.form.get('model_name')

        if data_name == "iris":
            from sklearn.datasets import load_iris
            data = load_iris()
        elif data_name == "wine":
            from sklearn.datasets import load_wine
            data = load_wine()
        else:
            return render_template("index.html", err1="Please select iris or wine.")
        
        descr = data.DESCR.replace("\n", "<br>").replace(" ", "&nbsp;")
        
        if model_name == "dt":
            model = tree.DecisionTreeClassifier()
        elif model_name == "svc":
            model = SVC(kernel='linear', probability=True)
        else:
            return render_template("index.html", err2="Please select dt or svc.")
        
        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=2022)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        fig = draw_confusion_matrix(confusion_matrix(y_test, y_pred), data.target_names)
        # グラフ画像を保存
        filepath = "./graph/" + datetime.now().strftime("%Y%m%d%H%M%S_") + "graph.png"
        fig.savefig(filepath)

        accuracy = round(accuracy_score(y_test, y_pred), 3) * 100
        report = classification_report(y_test, y_pred, target_names=data.target_names, digits=3, output_dict=True)
        pd.options.display.precision = 3
        report = pd.DataFrame(report).T.drop("accuracy")
        report_html = report.to_html()

        return render_template("index.html", 
                                data_name = data_name,
                                model_name = model_name,
                                filepath=filepath,
                                accuracy = accuracy,
                                report = report_html,
                                descr = descr,
                                )

if __name__ == '__main__':
    app.run(debug=True,  host='0.0.0.0', port=9999) # ポートの変更