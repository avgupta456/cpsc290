import pickle

def load_matrix(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

def load_data(dataset):
    path = "./datasets/" + str(dataset) + "/processed"
    train = load_matrix(path + '/train.p')
    test = load_matrix(path + '/test.p')
    val = load_matrix(path + '/val.p')
    return train, test, val
