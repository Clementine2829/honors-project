import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from  my_models.SupportVectorMachineClass.SupportVectorMachine import SVM


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

    X = df[['FlowDuration', 'TotalFwdPackets', 'TotalFwdPackets', 'TotalBackwardPackets',
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

    return round(accuracy, 2)

# plot results
def get_hyperplane(x, w, b, offset):
    return (-w[0] * x - b + offset) / w[1]


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
    predictions = clf.predict(X_test)

    _accuracy = str(accuracy(y_test, predictions)) + "%"
    return _accuracy, clf, X_train, X_test, y_train, y_test


#size = 6
#size = 4
#accuracy, clf, X_train, X_test, y_train, y_test = predict(size)
#print("Accuracy: " + str(accuracy))
#plot_graph(clf, X_train, X_test, y_train, y_test)
