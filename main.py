from flask import Flask, render_template

from my_models.SupportVectorMachineClass.svm_main import accuracy

app = Flask(__name__)


@app.route("/home")
@app.route("/")
def home():
    return render_template("backend-prototype.html")


@app.route("/client")
def client():
    return render_template("frontend-prototype.html")

# JINJA2
def mygnb(size):
    import os, sys
    my_lib_path = os.path.abspath('c:/Users/CLEMENTINE/Desktop/pythonProject/project/my_models/NaiveBayesGuassianPractice1/nb_main.py')
    sys.path.append(my_lib_path)

    from my_models.NaiveBayesGuassianPractice1.nb_main import load_main

    accuracy = load_main(size)
    print("accuracy: " + str(accuracy))
   
    #return render_template("frontend-prototype.html", ID=accuracy)
    return {"accuracy": accuracy}

def mydnn(size):

    import os, sys
    my_lib_path = os.path.abspath('c:/Users/CLEMENTINE/Desktop/pythonProject/project/my_models/NeuralNetworkClass/nn_main.py')
    sys.path.append(my_lib_path)

    from my_models.NeuralNetworkClass.nn_main import load_main, print_confusion_matrix

    accuracy, pred_values, truth_values = load_main(size)
    print_confusion_matrix(pred_values, truth_values)
    
    #return render_template("frontend-prototype.html", ID=accuracy)
    return {"accuracy": accuracy}


def mysvm(size):
    import os, sys
    my_lib_path = os.path.abspath('c:/Users/CLEMENTINE/Desktop/pythonProject/project/my_models/SupportVectorMachineClass/svm_main.py')
    sys.path.append(my_lib_path)

    from my_models.SupportVectorMachineClass.svm_main import predict, plot_graph

    accuracy, clf, X_train, X_test, y_train, y_test = predict(size)
    print("Accuracy: " + str(accuracy))
    plot_graph(clf, X_train, X_test, y_train, y_test)
    
    #return render_template("frontend-prototype.html", ID=accuracy)
    return {"accuracy": accuracy}

@app.route("/<dataset>/<model>")
def test(model, dataset):
    size = 6
    if dataset == "kdd":
        size = 4
    if model == "nb":
        return mygnb(size)
    elif model == "nn":
        return mydnn(size)
    elif model == "svm":
        return mysvm(size)
    else:
        return ""


if __name__ == "__main__":
    app.run(debug=True)
