from array import array
import datetime
import json
from unittest import result
import numpy as np
import pandas as pd
import joblib
from pickle import load

from my_models.NaiveBayesGuassianPractice1.GuassianClass import GaussianNB
#from GuassianClass import GaussianNB

def accuracy_score(y_true, y_pred):
    # return (y_true - y_pred) / len(y_true)
    total = sum(y_pred == y_true)
    return round(float(total) / float(len(y_true)) * 100, 2)


def train_test_split(x, y, test_size=0.25, random_state=None):
    """ partioning the data into train and test sets """

    x_test = x.sample(frac=test_size, random_state=random_state)
    y_test = y[x_test.index]

    x_train = x.drop(x_test.index)
    y_train = y.drop(y_test.index)

    return x_train, x_test, y_train, y_test


def kdd_train_pre_processing(df):
    """ partioning data into features and target """
    df[['class']] = np.where(df[['class']] == 'normal', 1, 0)
    df.loc[df["protocol_type"] == "tcp", "protocol_type"] = 1
    df.loc[df["protocol_type"] == "udp", "protocol_type"] = 2
    df.loc[df["protocol_type"] == "icmp", "protocol_type"] = 3

    X = df[['duration', 'protocol_type', 'src_bytes', 'dst_bytes']]
    y = df[df.columns[-1]]

    return X, y


def cicids_train_pre_processing(df):
    """ partioning data into features and target """
    # remove special character
    df[['Label']] = np.where(df[['Label']] == 'BENIGN', 1, 0)

    X = df[['FlowDuration', 'FlowDuration', 'TotalFwdPackets', 'TotalBackwardPackets',
            'TotalLengthofFwdPackets', 'TotalLengthofBwdPackets']]
    y = df[df.columns[-1]]

    return X, y


def test_data(x, y, test_size=0.25, random_state=None):
    """ partioning the data into train and test sets """

    x = x.sample(frac=test_size, random_state=random_state)

    return x, y


def kdd_test_pre_processing(df):
    """ partioning data into features and target """
    df.loc[df["protocol_type"] == "tcp", "protocol_type"] = 1
    df.loc[df["protocol_type"] == "udp", "protocol_type"] = 2
    df.loc[df["protocol_type"] == "icmp", "protocol_type"] = 3

    X = df[['duration', 'protocol_type', 'src_bytes', 'dst_bytes']]
    y = df[df.columns[-1]]

    return X, y


def train_data(x, y, test_size=0.25, random_state=None):
    """ partioning the data into train and test sets """

    x = x.sample(frac=test_size, random_state=random_state)
    y = y[x.index]

    x_train = x.drop(x.index)
    y_train = y.drop(y.index)

    return x, y


def test_data(x, y, test_size=0.25, random_state=None):
    """ partioning the data into train and test sets """

    x = x.sample(frac=test_size, random_state=random_state)

    return x, y


def dataset_selector(df, size):
    if size == 4:
        return kdd_train_pre_processing(df)
    else:
        return cicids_train_pre_processing(df)


def data_train(size):
    if size == 4:
        df_train = pd.read_csv("c:/Users/CLEMENTINE/Desktop/pythonProject/project/my_models/data/train_data.csv")
        # df_test = pd.read_csv("c:/Users/CLEMENTINE/Desktop/pythonProject/project/my_models/data/test_data.csv")
        return df_train
    else:
        #df_train = pd.read_csv("c:/Users/CLEMENTINE/Desktop/pythonProject/project/my_models/data/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
        df_train = pd.read_csv("c:/Users/CLEMENTINE/Desktop/pythonProject/project/my_models/data/Wednesday-workingHours.pcap_ISCX.csv")

        df_train.columns = df_train.columns.str.replace(' ', '')
        df_train.dropna(inplace=True)

        return df_train


def model_select(size, d):
    
    data = np.array(d["data"])

    data_to_file = "Naive Bayes Scanning data " +  str(datetime.datetime.now()) + "\n"
    if size != "4":
        mj = joblib.load("C:/Users/CLEMENTINE/Desktop/pythonProject/project/my_models/TraindModels/cicids_joblib_model")
        X_data = data.reshape(-1, 6)
    else:
        mj = joblib.load("C:/Users/CLEMENTINE/Desktop/pythonProject/project/my_models/TraindModels/nb_joblib_model")
        temp_data = data.reshape(-1, 5)
        X_data = []
        for x in range(len(temp_data)):
            j = list(temp_data[x])
            j.pop(3) # remove fwd length
            # j.pop(2) # remove bck length

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


def load_main(size):
    df_train = data_train(size)
    X_train, y_train = dataset_selector(df_train, size)

    # Split data into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=.8, random_state=0)

    gnb_clf = GaussianNB()
    gnb_clf.fit(X_train, y_train)

    if(size == 4):
        model_path = "C:/Users/CLEMENTINE/Desktop/pythonProject/project/my_models/TraindModels/nb_joblib_model"
        joblib.dump(gnb_clf, model_path)        
        model_nb = joblib.load(model_path)

    else:
        model_path = "C:/Users/CLEMENTINE/Desktop/pythonProject/project/my_models/TraindModels/cicids_joblib_model"
        joblib.dump(gnb_clf, model_path)
        model_nb = joblib.load(model_path)

    predict = model_nb.predict(X_test)

    accuracy = accuracy_score(y_test, predict)

    if(size == 4):
        f = open("C:/Users/CLEMENTINE/Desktop/pythonProject/project/nb_model.txt", "w")
        f.write("4")
        f.close()

    else:
        f = open("C:/Users/CLEMENTINE/Desktop/pythonProject/project/nb_model.txt", "w")
        f.write("6")
        f.close()

    f = open("C:/Users/CLEMENTINE/Desktop/pythonProject/project/nb_model_percentage.txt", "w")
    f.write(str(accuracy) + "%")
    f.close()

    return str(accuracy) + "%"

# size = 6
# size = 4
# accuracy = load_main(size)
# print("accuracy: " + str(accuracy))
