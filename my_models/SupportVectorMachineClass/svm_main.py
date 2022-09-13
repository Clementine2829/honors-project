import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# the below are used for plotting graphs 
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns

import joblib
from pickle import load

from  my_models.SupportVectorMachineClass.SupportVectorMachine import SVM
#from  SupportVectorMachine import SVM


def kdd_train_pre_processing(df):
    """ partioning data into features and target """
    df[['class']] = np.where(df[['class']] == 'normal', 1, 0)
    df.loc[df["protocol_type"] == "tcp", "protocol_type"] = 1
    df.loc[df["protocol_type"] == "udp", "protocol_type"] = 2
    df.loc[df["protocol_type"] == "icmp", "protocol_type"] = 3

    X = df[['duration', 'protocol_type', 'src_bytes', 'dst_bytes']]
    y = df[df.columns[-1]]

    #    print("Pre processing x")
    #    print(X)

    #    print("Pre processing y")
    #    print(y)

    return X, y


def cicids_train_pre_processing(df):
    """ partioning data into features and target """
    # remove special character
    df[['Label']] = np.where(df[['Label']] == 'BENIGN', 1, 0)

    X = df[['FlowDuration', 'FlowDuration', 'TotalFwdPackets', 'TotalBackwardPackets',
            'TotalLengthofFwdPackets', 'TotalLengthofBwdPackets']]
    y = df[df.columns[-1]]

    #print("Pre processing x")
    #print(df)
    #print("Pre processing y")
    #print(y)

    return X, y


def kdd_test_pre_processing(df):
    """ partioning data into features and target """
    df.loc[df["protocol_type"] == "tcp", "protocol_type"] = 1
    df.loc[df["protocol_type"] == "udp", "protocol_type"] = 2
    df.loc[df["protocol_type"] == "icmp", "protocol_type"] = 3

    X = df[['duration', 'protocol_type', 'src_bytes', 'dst_bytes']]
    y = df.add("class", axis=1)

    return X, y


def train_data(x, y, test_size=0.25, random_state=None):
    """ partioning the data into train and test sets """

    x_test = x.sample(frac=test_size, random_state=random_state)
    y_test = y[x_test.index]

    x_train = x.drop(x_test.index)
    y_train = y.drop(y_test.index)

    return x_train, x_test, y_train, y_test


def test_data(x, y, test_size=0.25, random_state=None):
    """ partioning the data into train and test sets """

    x = x.sample(frac=test_size, random_state=random_state)

    return x, y


def dataset_selector(size):
    if size == 4:
        df = pd.read_csv("c:/Users/CLEMENTINE/Desktop/pythonProject/project/my_models/data/train_data.csv")
        # df = pd.read_csv("c:/Users/CLEMENTINE/Desktop/pythonProject/project/my_models/data/test_data.csv")

        return kdd_train_pre_processing(df)
    else:
        df = pd.read_csv("c:/Users/CLEMENTINE/Desktop/pythonProject/project/my_models/data/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
        # df_train = pd.read_csv("c:/Users/CLEMENTINE/Desktop/pythonProject/project/my_models/data/Wednesday-workingHours.pcap_ISCX.csv")

        df.columns = df.columns.str.replace(' ', '')
        df.dropna(inplace=True)

        return cicids_train_pre_processing(df)


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true) * 100
    print("y_true")
    print(y_true)
    print("y_pred")
    print(y_pred)

    return round(accuracy, 2)

# plot results
def get_hyperplane(x, w, b, offset):
    return (-w[0] * x - b + offset) / w[1]


def model_select(size, data):
    
    f = open("C:/Users/CLEMENTINE/Desktop/pythonProject/project/kdd_model.txt", "w")
    f.write("svm")
    f.close()

    data_to_file = "SVM Scanning data " +  str(datetime.datetime.now()) + "\n"
    if size != 4:
        mj = joblib.load("C:/Users/CLEMENTINE/Desktop/pythonProject/project/my_models/TraindModels/cicids_joblib_model")
        X_data = data.reshape(-1, 5)
    else:
        mj = joblib.load("C:/Users/CLEMENTINE/Desktop/pythonProject/project/my_models/TraindModels/kdd_joblib_model")
        temp_data = data.reshape(-1, 5)
        X_data = []
        for x in range(len(temp_data)):
            j = list(temp_data[x])
            j.pop(3) # remove fwd length
            j.pop(2) # remove bck length

            X_data.append(j)

    predict = mj.predict(X_data)

    results = []
    percentage = 0
    for x in range(len(temp_data)):
        j = list(temp_data[x])
        if(predict[x] == 0):
            j.append(False)
        else:
            j.append(True)
            percentage += 1
        data_to_file += str(j) + "\n"
    
        results.append(j)

    percentage = percentage / len(temp_data) * 100     

    data_to_file += "Percentage: " + str(percentage) + "%\n"
    f = open("C:/Users/CLEMENTINE/Desktop/pythonProject/project/log.txt", "a")
    f.write(data_to_file)
    f.close()

    return results


def plot_graph(clf, X_train, X_test, y_train, y_test):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    plt.set_cmap('PiYG')
    plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', c=y_train, s=200, alpha=0.50)
    plt.scatter(X_test[:, 0], X_test[:, 1], marker="x", c=y_test, s=200, alpha=0.50)

    x0_1 = np.amin(X_train[:, 0])
    x0_2 = np.amax(X_train[:, 0])

    x1_1 = get_hyperplane(x0_1, clf.w, clf.b, 0)
    x1_2 = get_hyperplane(x0_2, clf.w, clf.b, 0)

    x1_1_m = get_hyperplane(x0_1, clf.w, clf.b, -1)
    x1_2_m = get_hyperplane(x0_2, clf.w, clf.b, -1)

    x1_1_p = get_hyperplane(x0_1, clf.w, clf.b, 1)
    x1_2_p = get_hyperplane(x0_2, clf.w, clf.b, 1)

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "-", c='k', lw=1, alpha=0.9)
    ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "--", c='grey', lw=1, alpha=0.8)
    ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "--", c='grey', lw=1, alpha=0.8)

    x1_min = np.amin(X_train[:, 1])
    x1_max = np.amax(X_train[:, 1])
    ax.set_ylim([x1_min - 3, x1_max + 3])

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    plt.show()

def roc_curve(y, pred):
    #print("Y")
    #print(y)
    #print("pred")
    #print(pred)

    #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.RocCurveDisplay.html#sklearn.metrics.RocCurveDisplay
    fpr, tpr, thresholds = metrics.roc_curve(np.array(y), np.array(pred), pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='SVM ROC Curve')
    
    display.plot()
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    print("confusiom matrix values")

    print(len(y_true))
    print(len(y_pred))
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(y_pred)):
        if i < 150:
            print("Truth: " + str(y_true[i]) + " Pred: " + str(y_pred[i]))

        if y_true[i] == 1 and y_pred[i] == 1:
            tp += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            fn += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            fp += 1
        elif y_true[i] == 0 and y_pred[i] == 0:
            tn += 1

    print("TP: " + str(tp))
    print("TN: " + str(tn))
    print("FP: " + str(fp))
    print("FN: " + str(fn))


    confusion_matrix(y_true, y_true)
    cf_matrix = confusion_matrix(y_true, y_true)
    print(cf_matrix)
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

    ax.set_title('Seaborn Confusion Matrix\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')

    # Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['Normal', 'Intrusion'])
    ax.yaxis.set_ticklabels(['Normal', 'Intrusion'])

    # Display the visualization of the Confusion Matrix.
    plt.show()

def rearange_data(x):


    return x

def model_select(size, d):

    data = np.array(d["data"])
    temp_data = data.reshape(-1, 6)

    print("Size ")
    print(size)

    data = np.array(d["data"])
    print("data")
    print(data)

    data_to_file = "Suppory Vector Machine Scanning data " +  str(datetime.datetime.now()) + "\n"
    if size != "4":
        mj = joblib.load("C:/Users/CLEMENTINE/Desktop/pythonProject/project/my_models/TraindModels/cicids_joblib_model")
        X_data = []
        for x in range(len(temp_data)):
            j = rearange_data(list(temp_data[x]))
            print(j)
            X_data.append(j)

    else:
        mj = joblib.load("C:/Users/CLEMENTINE/Desktop/pythonProject/project/my_models/TraindModels/svm_joblib_model")
        X_data = []
        for x in range(len(temp_data)):
            j = list(temp_data[x])
            j.pop(3) # remove fwd length
            j.pop(2) # remove bck length

            X_data.append(j)

    predict = mj.predict(X_data)

    results = []
    percentage = 0
    for x in range(len(temp_data)):
        j = list(temp_data[x])
        if(predict[x] == 0):
            j.append(False)
            percentage += 1
        else:
            j.append(True)
        data_to_file += str(j) + "\n"
    
        results.append(j)

    percentage = percentage / len(temp_data) * 100     

    data_to_file += "Percentage: " + str(percentage) + "%\n"
    f = open("C:/Users/CLEMENTINE/Desktop/pythonProject/project/log.txt", "a")
    f.write(data_to_file)
    f.close()

    return results, round(percentage, 2)



def predict(size):
    X_train, y_train = dataset_selector(size)

    # Split data into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_data(X_train, y_train, test_size=0.8, random_state=0)

    # remove labels and convert to array
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    X_train = X_train.astype('float32')

    clf = SVM(n_iters=1000)
    clf.fit(X_train, y_train)

    if(size == 4):
        model_path = "C:/Users/CLEMENTINE/Desktop/pythonProject/project/my_models/TraindModels/svm_joblib_model"
        joblib.dump(clf , model_path)        
        model_svm = joblib.load(model_path)

    else:
        model_path = "C:/Users/CLEMENTINE/Desktop/pythonProject/project/my_models/TraindModels/cicids_joblib_model"
        joblib.dump(clf, model_path)
        model_svm = joblib.load(model_path)

    predictions = model_svm.predict(X_test)

    print("X_data")
    print(X_test)

    print("Y Data")
    print(y_test)

    print("Predictions")
    print(predictions)


    _accuracy = str(accuracy(y_test, predictions)) + "%"

    if(size == 4):
        f = open("C:/Users/CLEMENTINE/Desktop/pythonProject/project/svm_model.txt", "w")
        f.write("4")
        f.close()

    else:
        f = open("C:/Users/CLEMENTINE/Desktop/pythonProject/project/svm_model.txt", "w")
        f.write("6")
        f.close()

    f = open("C:/Users/CLEMENTINE/Desktop/pythonProject/project/svm_model_percentage.txt", "w")
    f.write(str(_accuracy))
    f.close()

    return _accuracy, clf, X_train, X_test, y_train, y_test, predictions


# size = 6
# #size = 4
# accuracy, clf, X_train, X_test, y_train, y_test, predictions = predict(size)
# print("Accuracy: " + str(accuracy))
# plot_graph(clf, X_train, X_test, y_train, y_test)
# print("roc curve")
# roc_curve(predictions, y_test)
# # print("confusion matrix")
# # y_test = [1, 1, 0, 0, 1, 1, 1, 1, 1 ,1]
# # predictions = [1 ,1 ,1, 1 ,1, 1, 1, 1, 1 ,0]

# plot_confusion_matrix(y_test, predictions)
