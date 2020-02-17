import argparse
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset', help="which dataset to use", required=True)
    parser.add_argument('-p', '--n_people', help="number of people to add until", type=int, default=50)
    parser.add_argument('-f', '--n_features', help="number of features being used, including name", type=int, default=4)

    return parser.parse_args()

def import_data(dataset):
	dataset = str(dataset)
	path = "../datasets/" # this should get you to these files, edit if it doesn't
	return np.genfromtxt(path + dataset + "/DS_utils/features.txt", dtype = 'str')

def save_data(dataset, data):
    dataset = str(dataset)
    path = "../datasets/"
    np.savetxt(path + dataset + "/DS_utils/features_fake.txt", data, fmt = '%s')

if __name__ == "__main__":
    args = get_args()
    dataset = args.dataset
    n_people = args.n_people
    n_features = args.n_features

    print("importing data..")
    posArr = import_data(dataset)
    print("data imported")

    print("adding 'fake' data..")
    add_size = (n_features*n_people+1)-posArr.shape[1]
    posArrAdd = np.full(fill_value='fake', shape=(posArr.shape[0], add_size))
    posArrNew = np.append(posArr, posArrAdd, axis=1)
    print("'fake' data added")

    print("saving augmented dataset")
    save_data(dataset, posArrNew)
    print("augmented dataset saved")
