import numpy as np
import re

def load_features(features_path, groups_path, max_people, features):
    file1 = open(groups_path, "r")
    lines1 = file1.readlines()
    points = len(lines1)

    num_features = features[0] + features[1] + features[2]
    X = np.empty(shape=(points, 1+max_people*(num_features+1)), dtype="U25")
    Y = np.empty(shape=(points, 1+2*max_people), dtype="U25")

    file2 = open(features_path)
    lines2 = file2.readlines()

    pos = 0
    time = float(lines1[pos].split()[0])

    for i in range(len(lines2)):
        line = lines2[i].split()
        if abs(float(line[0])-time)<0.05 and pos<points:
            X[pos][0] = time
            people = int((len(line)-1)/(num_features+1))

            for j in range(people):
                X[pos][(num_features+1)*j+1:(num_features+1)*(j+1)+1] = line[(num_features+1)*j+1:(num_features+1)*(j+1)+1]

            for j in range(people, max_people):
                X[pos][(num_features+1)*j+1] = "fake"
                X[pos][(num_features+1)*j+2:(num_features+1)*(j+1)+1] = [0]*num_features

            Y[pos][0] = time

            groups_arr = re.split(" < | > < ", lines1[pos].strip())
            ids_arr = {} #dictionary
            for j in range(1,len(groups_arr)):
                ids = groups_arr[j].split()
                if(j==len(groups_arr)-1): ids = ids[:-1]
                for id in ids: ids_arr[id] = j

            for j in range(people):
                Y[pos][2*j+1] = X[pos][(num_features+1)*j+1]
                Y[pos][2*j+2] = ids_arr[Y[pos][2*j+1]]

            for j in range(people, max_people):
                Y[pos][2*j+1] = "fake"
                Y[pos][2*j+2] = -1

            pos = min(pos+1, points-1)
            time = float(lines1[pos].split()[0])

    return X, Y

features_path = "./datasets/cocktail/raw/features_expanded.txt"
groups_path = "./datasets/cocktail/raw/groups.txt"

max_people = 20 #max people possible

f_pos_xy = 2 #space stored for x, y
f_pos_angle = 2 #space stored for angles related to position
f_normal = 0 #space stored for all other features
features = [f_pos_xy, f_pos_angle, f_normal]
null = [0, 0, 0, 0, 0, 0]

X, Y = load_features(features_path, groups_path, max_people, features)
print(X)
print(Y)
