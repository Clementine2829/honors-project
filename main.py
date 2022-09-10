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

    accuracy, clf, X_train, X_test, y_train, y_test, predictions = predict(size)
    print("Accuracy: " + str(accuracy))
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


def load_model(size, data):
    if size == 4:
        f = open("C:/Users/CLEMENTINE/Desktop/pythonProject/project/kdd_model.txt", "r")
        model = f.readline()
        f.close() 
    else:
        f = open("C:/Users/CLEMENTINE/Desktop/pythonProject/project/cicids_model.txt", "r")
        model = f.readline()
        f.close()

    if (model == "nb"):
        from run_model import model_select_nb

        # return a 2D array
        model = "Naive Bayes"
        results = {'data': str(model_select_nb(size, data)), "percentage": "80%", "model": model}
        return results

    elif (model == "nn"):
        from run_model import model_select_nn

        # return a 2D array
        model = "Neural Network"
        results = {'data': str(model_select_nn(size, data)), "percentage": "80%", "model": model}
        return results

    elif (model == "svm"):
        from run_model import model_select_svm

        # return a 2D array
        model = "Support Vector Machine"
        results = {'data': str(model_select_svm(size, data)), "percentage": "80%", "model": model}
        return results

    else:
        return {"message": "Internal error occured."}


@app.route("/client/<model>", methods = ['POST'])
def test_model(model):
    data = request.json
    # print("model")
    # print(model)
    # print(data)

    # if (model == "nb"):
    size = 4
    # data = np.array([1,1,1517,332,44,0,1,334,0,45,0,1,314,810,100,0,1,0,0,0,0,1,0,0,122,0,1,0,0,1584,0,1,458,6452,100,5,1,11895,1359,652,1,1,851,325,2355,0,1,0,0,0,0,0,0,0,755,1,0,33,45,66,2,0,3,0,0,5,1,650,70,230,0,1,100,455,1000])

    return load_model(size, data)



if __name__ == "__main__":
    app.run(debug=True)
