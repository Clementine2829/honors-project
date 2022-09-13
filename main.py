from unittest import result
from flask import Flask, render_template, request

from my_models.SupportVectorMachineClass.svm_main import accuracy, roc_curve, plot_confusion_matrix
from run_model import model_select_nb

app = Flask(__name__)


@app.route("/home")
@app.route("/")
def home():
    return render_template("backend-prototype.html")


@app.route("/client")
def client():


    try:      
        f = open("C:/Users/CLEMENTINE/Desktop/pythonProject/project/nb_model_percentage.txt", "r")
        nb = f.readline() # this file has the size of the dataset used
        f.close()
    except FileNotFoundError:
        nb = "NA"

    try:      
        f = open("C:/Users/CLEMENTINE/Desktop/pythonProject/project/nn_model_percentage.txt", "r")
        nn = f.readline() # this file has the size of the dataset used
        f.close()
    except FileNotFoundError:
        nn = "NA"

    try:      
        f = open("C:/Users/CLEMENTINE/Desktop/pythonProject/project/svm_model_percentage.txt", "r")
        svm = f.readline() # this file has the size of the dataset used
        f.close() 
    except FileNotFoundError:
        svm = "NA"

    return render_template("frontend-prototype.html", NB_MODEL = nb, NN_MODEL = nn, SVM_MODEL = svm)

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

    accuracy, clf, X_train, X_test, y_train, y_test, predictions = predict(size)
    # print("Accuracy: " + str(accuracy))
    plot_graph(clf, X_train, X_test, y_train, y_test)
    roc_curve(predictions, y_test)
    plot_confusion_matrix(y_test, predictions)

    
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


def load_model(model, data):
    if model == "nb":
        f = open("C:/Users/CLEMENTINE/Desktop/pythonProject/project/nb_model.txt", "r")
        size = f.readline() # this file has the size of the dataset used
        f.close() 
    elif model == "nn":
        f = open("C:/Users/CLEMENTINE/Desktop/pythonProject/project/nn_model.txt", "r")
        size = f.readline() # this file has the size of the dataset used
        f.close() 
    elif model == "svm":
        f = open("C:/Users/CLEMENTINE/Desktop/pythonProject/project/svm_model.txt", "r")
        size = f.readline() # this file has the size of the dataset used
        f.close() 
        
    if (model == "nb"):
        from run_model import model_select_nb

        # return a 2D array
        model = "Naive Bayes"
        data, percentage = model_select_nb(size, data)
        results = {'data': str(data), "percentage": str(percentage) + "%", "model": model}
        return results

    elif (model == "nn"):
        from run_model import model_select_nn

        # return a 2D array
        model = "Neural Network"
        data, percentage = model_select_nn(size, data)
        results = {'data': str(data), "percentage": str(percentage) + "%", "model": model}
        return results

    elif (model == "svm"):
        from run_model import model_select_svm

        # return a 2D array
        model = "Support Vector Machine"
        data, percentage = model_select_svm(size, data)
        results = {'data': str(data), "percentage": str(percentage) + "%", "model": model}
        return results

    else:
        return {"message": "Internal error occured."}


@app.route("/client/<model>", methods = ['POST'])
def test_model(model):
    data = request.json
    return load_model(model, data)



if __name__ == "__main__":
    app.run(debug=True)
