import sys
from flask import Flask, render_template

app = Flask(__name__)


@app.route("/home")
@app.route("/")
def home():
    return render_template("backend-prototype.html")


@app.route("/client")
def client():
    return render_template("frontend-prototype.html")

# JINJA2
@app.route("/gnb")
def mygnb():
    import os, sys
    my_lib_path = os.path.abspath('c:/Users/CLEMENTINE/Desktop/pythonProject/project/my_models/NaiveBayesGuassianPractice1/nb_main.py')
    sys.path.append(my_lib_path)

    from my_models.NaiveBayesGuassianPractice1.nb_main import load_main

    # size = 6
    size = 4

    accuracy = load_main(size)
    print("accuracy: " + str(accuracy))
    return render_template("frontend-prototype.html", ID=accuracy)

@app.route("/dnn")
def mydnn():

    import os, sys
    my_lib_path = os.path.abspath('c:/Users/CLEMENTINE/Desktop/pythonProject/project/my_models/NeuralNetworkClass/nn_main.py')
    sys.path.append(my_lib_path)

    from my_models.NeuralNetworkClass.nn_main import load_main, print_confusion_matrix

    #size = 6
    size = 4
    accuracy, pred_values, truth_values = load_main(size)
    print(accuracy)
    #print("truth values")
    #print(truth_values)
    #print("pred values")
    #print(pred_values)

    #print_confusion_matrix(pred_values, truth_values)
    
    return render_template("frontend-prototype.html", ID=accuracy)

@app.route("/svm")
def mysvm():

    import os, sys
    my_lib_path = os.path.abspath('c:/Users/CLEMENTINE/Desktop/pythonProject/project/my_models/SupportVectorMachineClass/svm_main.py')
    sys.path.append(my_lib_path)

    from my_models.SupportVectorMachineClass.svm_main import predict, plot_graph

    # size = 6
    size = 4

    accuracy, clf, X_train, X_test, y_train, y_test = predict(size)
    print("Accuracy: " + str(accuracy))
    plot_graph(clf, X_train, X_test, y_train, y_test)
    
    return render_template("frontend-prototype.html", ID=accuracy)


if __name__ == "__main__":
    app.run(debug=True)
