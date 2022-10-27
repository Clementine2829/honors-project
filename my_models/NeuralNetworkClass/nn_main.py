import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from my_models.NeuralNetworkClass.n_network import MyNeuralNetwork

from my_models.NeuralNetworkClass.fc_layer import FCLayer
from my_models.NeuralNetworkClass.activation_layer import ActivationLayer

from keras.utils import to_categorical

from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns

import joblib
from pickle import load

def sigmoid(x):
    z = np.exp(-x)
    sig = 1 / (1 + z)
    return sig


def der_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


# loss function and its derivative
def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


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


def kdd_test_pre_processing(df):
    """ partioning data into features and target """
    df.loc[df["protocol_type"] == "tcp", "protocol_type"] = 1
    df.loc[df["protocol_type"] == "udp", "protocol_type"] = 2
    df.loc[df["protocol_type"] == "icmp", "protocol_type"] = 3

    X = df[['duration', 'protocol_type', 'src_bytes', 'dst_bytes']]
    y = df.add("class", axis=1)

    return X, y


def cicids_train_pre_processing(df):
    """ partioning data into features and target """
    # remove special character
    df[['Label']] = np.where(df[['Label']] == 'BENIGN', 1, 0)

    X = df[['DestinationPort', 'FlowDuration', 'TotalFwdPackets', 'TotalBackwardPackets',
            'TotalLengthofFwdPackets', 'TotalLengthofBwdPackets']]
    y = df[df.columns[-1]]

    print("Data ")
    print(X)
    print(y)
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


def dataset_selector(df, size):
    if size == 4:
        return kdd_train_pre_processing(df)
    else:
        return cicids_train_pre_processing(df)


def pre_process_model(df_train, size):
    X_train, y_train = dataset_selector(df_train, size)

    # Split data into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_data(X_train, y_train, test_size=.8, random_state=0)

    # remove labels and convert to array
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # print(X_train)
    X_train = X_train.reshape(-1, 1, size)
    X_train = X_train.astype('float32')
    X_train /= 255

    y_train = to_categorical(y_train)

    X_test = X_test.reshape(-1, 1, size)
    X_test = X_test.astype('float32')
    X_test /= 255

    y_test = to_categorical(y_test)

    return X_train, y_train, X_test, y_test


def data_train(size):
    if size == 4:
        df_train = pd.read_csv("c:/Users/CLEMENTINE/Desktop/pythonProject/project/my_models/data/train_data.csv")
        #df_test = pd.read_csv("c:/Users/CLEMENTINE/Desktop/pythonProject/project/my_models/data/test_data.csv")
        return df_train
    else:
        df_train = pd.read_csv("c:/Users/CLEMENTINE/Desktop/pythonProject/project/my_models/data/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
        # df_train = pd.read_csv("c:/Users/CLEMENTINE/Desktop/pythonProject/project/my_models/data/Wednesday-workingHours.pcap_ISCX.csv")

        df_train.columns = df_train.columns.str.replace(' ', '')
        df_train.dropna(inplace=True)

        return df_train


def model_select(size, d):
    data = np.array(d["data"])
    temp_data = data.reshape(-1, 6)

    data_to_file = "Neural Network Scanning data " +  str(datetime.datetime.now()) + "\n"
    if size != "4":
        # mj = joblib.load("C:/Users/CLEMENTINE/Desktop/pythonProject/project/my_models/TraindModels/cicids_joblib_model")
        mj = joblib.load("C:/Users/CLEMENTINE/Desktop/pythonProject/project/my_models/TraindModels/nn_joblib_model")
        X_data = []
        for x in range(len(temp_data)):
            j = list(temp_data[x])
            X_data.append(j)
        
    else:
        mj = joblib.load("C:/Users/CLEMENTINE/Desktop/pythonProject/project/my_models/TraindModels/nn_joblib_model")
        X_data = []
        for x in range(len(temp_data)):
            j = list(temp_data[x])
            j.pop(3) # remove fwd length
            j.pop(2) # remove bck length

            X_data.append(j)
        
    predict = mj.predict(X_data)

    arr_predict = []
    for x in range(len(predict)):
        d = list(predict[x])
        if (d[0][0] < d[0][1]):
            arr_predict.append(0)
        else:
            arr_predict.append(1)

    results = []
    percentage = 0
    for x in range(len(temp_data)):
        j = list(temp_data[x])
        if(arr_predict[x] == 0):
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
 

def load_main(size):
    net = MyNeuralNetwork()
    net.add(FCLayer(4, size * 4))
    net.add(ActivationLayer(sigmoid, der_sigmoid))
    net.add(FCLayer(size * 4, size * 2))
    net.add(ActivationLayer(sigmoid, der_sigmoid))
    net.add(FCLayer(size * 2, 2))
    net.add(ActivationLayer(sigmoid, der_sigmoid))

    net.use(mse, mse_prime)

    df_train = data_train(size)
    X_train, y_train, X_test, y_test = pre_process_model(df_train, size)

    print("Data again")
    print(X_train[0:])
    print(y_train[0:])
    # net.fit(X_train[0:], y_train[0:], epochs=X_train.size, learning_rate=0.1)
    net.fit(X_train[0:], y_train[0:], epochs=5, learning_rate=0.8)


    # if(size == 4):
    model_path = "C:/Users/CLEMENTINE/Desktop/pythonProject/project/my_models/TraindModels/nn_joblib_model"
    joblib.dump(net, model_path)        
    model_nn = joblib.load(model_path)

    # else:
    #     model_path = "C:/Users/CLEMENTINE/Desktop/pythonProject/project/my_models/TraindModels/cicids_joblib_model"
    #     joblib.dump(net, model_path)
    #     model_nn = joblib.load(model_path)


    out = model_nn.predict(X_test[0:])

    accuracy, pred_values, truth_values = net.get_accuracy(out, y_test[0:])

    if(size == 4):
        f = open("C:/Users/CLEMENTINE/Desktop/pythonProject/project/nn_model.txt", "w")
        f.write("4")
        f.close()

    else:
        f = open("C:/Users/CLEMENTINE/Desktop/pythonProject/project/nn_model.txt", "w")
        f.write("6")
        f.close()
    
    f = open("C:/Users/CLEMENTINE/Desktop/pythonProject/project/nn_model_percentage.txt", "w")
    f.write(str(accuracy) + "%")
    f.close()
    

    return str(accuracy) + "%", pred_values, truth_values


def print_confusion_matrix(truth_values, pred_values):
    cf_matrix = confusion_matrix(truth_values, pred_values)
    print(cf_matrix)
    ax = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.1%', cmap='YlGnBu')

    ax.set_title('Seaborn Confusion Matrix\n\n')
    ax.set_xlabel('\nPredicted')
    ax.set_ylabel('Actual')

    # Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['Normal', 'Intrusion'])
    ax.yaxis.set_ticklabels(['Normal', 'Intrusion'])

    # Display the visualization of the Confusion Matrix.
    plt.show()

def roc_curve(y, pred):

    fpr, tpr, thresholds = metrics.roc_curve(np.array(y), np.array(pred), pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='SVM ROC Curve')
    
    display.plot()
    plt.show()

#size = 6
#size = 4
#accuracy, pred_values, truth_values = load_main(size)
#print(accuracy)
#print("truth values")
#print(truth_values)
#print("pred values")
#print(pred_values)

#print_confusion_matrix(pred_values, truth_values)
