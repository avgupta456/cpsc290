import numpy as np
import random
import re

from helper import utils

def load_features(max_people, features, raw_path, viz_path):
    file1 = open(raw_path + "/groups.txt", "r")
    lines1 = file1.readlines()
    points = len(lines1)

    num_features = features[0] + features[1] + features[2]
    X = np.empty(shape=(points, 1+max_people*(num_features+1)), dtype="U50")
    Y = np.empty(shape=(points, 1+2*max_people), dtype="U50")

    file2 = open(raw_path + "/features.txt")
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

    file1.close()
    file2.close()

    np.savetxt(viz_path + "/X.txt", X, fmt='%s')
    np.savetxt(viz_path + "/Y.txt", Y, fmt='%s')

    return X, Y

def process_frame(X_frame, Y_frame, max_people, features, func=utils.transform):
    pre_features = features[0] + features[1] + features[2]
    post_features = features[0] + 2*features[1] + features[2]

    people = int((np.where(X_frame=="fake")[0][0]-1)/(pre_features+1))
    points = people*(people-1)

    pre_transform = np.empty(shape=(1, 1+max_people*pre_features), dtype="U50")
    X = np.empty(shape=(points, 1+max_people*post_features), dtype="U50")
    Y = np.empty(shape=(points, 2), dtype="U50")


    combinations = [[p1, p2] for p1 in range(people) for p2 in range(people) if p1!=p2]
    pos = 0

    for j, k in combinations:
        pre_transform[0][0] = X_frame[0]
        count = 0

        for m in range(people):
            if(m!=j and m!=k):
                pre_transform[0][pre_features*count+1:pre_features*(count+1)+1] = X_frame[(pre_features+1)*m+2:(pre_features+1)*(m+1)+1]
                count += 1

        for m in range(people, max_people):
            rand_p = random.randint(0, people-3) #includes final number
            pre_transform[0][pre_features*count+1:pre_features*(count+1)+1] = pre_transform[0][pre_features*rand_p+1:pre_features*(rand_p+1)+1]
            count += 1

        pre_transform[0][pre_features*count+1:pre_features*(count+1)+1] = X_frame[(pre_features+1)*j+2:(pre_features+1)*(j+1)+1]
        pre_transform[0][pre_features*(count+1)+1:pre_features*(count+2)+1] = X_frame[(pre_features+1)*k+2:(pre_features+1)*(k+1)+1]

        post_transform = func(pre_transform, j, k, max_people, features)
        affinity = 1 if Y_frame[2*(j+1)]==Y_frame[2*(k+1)] else 0

        X[pos] = post_transform
        Y[pos] = [X[pos][0], affinity]
        pos += 1

    return X, Y, pos

def build_dataset(X_old, Y_old, max_people, features, clean_path):
    pre_features = features[0] + features[1] + features[2]
    post_features = features[0] + 2*features[1] + features[2]

    times = np.empty(shape=(2*X_old.shape[0], 1), dtype="U50")
    times_pos = 0

    points = 0
    for i in range(len(X_old)):
        people = int((np.where(X_old[i]=="fake")[0][0]-1)/(pre_features+1))
        points += 2*people*(people-1)

    X = np.empty(shape=(points, 1+max_people*post_features), dtype="U50")
    Y = np.empty(shape=(points, 2), dtype="U50")

    pos = 0
    for i in range(len(X_old)):
        X_temp, Y_temp, points_temp = process_frame(X_old[i], Y_old[i], max_people, features, utils.transform)
        X[pos:pos+points_temp], Y[pos:pos+points_temp], times[times_pos] = X_temp, Y_temp, pos
        pos, times_pos = pos + points_temp, times_pos + 1

        X_temp, Y_temp, points_temp = process_frame(X_old[i], Y_old[i], max_people, features, utils.augment)
        X[pos:pos+points_temp], Y[pos:pos+points_temp], times[times_pos] = X_temp, Y_temp, pos
        pos, times_pos = pos + points_temp, times_pos + 1

    np.savetxt(clean_path + "/coordinates.txt", X, fmt='%s')
    np.savetxt(clean_path + "/affinities.txt", Y, fmt='%s')
    np.savetxt(clean_path + "/timechanges.txt", times, fmt='%s')

    return X, Y, times

def save_dataset(X, Y, times, max_people, features, processed_path):
    new_times = []
    for time in times:
        new_times.append(float(time[0]))

    post_features = features[0] + 2*features[1] + features[2]
    points = X.shape[0]

    X_group = np.zeros(shape=(points, 1, max_people-2, post_features))
    X_pairs = np.zeros(shape=(points, 1, 2, post_features))
    Y_new = np.zeros(shape=(points, 1), dtype=np.int8)

    for i in range(points):
        X_group[i][0] = np.reshape(X[i][1:-2*post_features], newshape=(max_people-2, post_features))
        X_pairs[i][0] = np.reshape(X[i][-2*post_features:], newshape=(2, post_features))
        Y_new[i][0] = int(Y[i][1])

    test = int(0.20*points)
    while(test not in new_times): test-=1
    test_index = new_times.index(test)
    times_test = new_times[:test_index]

    train = int(0.90*points)
    while(train not in new_times): treain -=1
    train_index = new_times.index(train)
    times_train = new_times[test_index:train_index]
    for i in range(len(times_train)): times_train[i]-=test

    val = points
    times_val = new_times[train_index:]
    for i in range(len(times_val)): times_val[i]-=train

    X_group_test = X_group[:test]
    X_group_train = X_group[test:train]
    X_group_val = X_group[train:val]

    X_pairs_test = X_pairs[:test]
    X_pairs_train = X_pairs[test:train]
    X_pairs_val = X_pairs[train:val]

    Y_test = Y_new[:test]
    Y_train = Y_new[test:train]
    Y_val = Y_new[train:val]

    utils.dump(processed_path + '/test.p', ([X_group_test, X_pairs_test], Y_test, times_test))
    utils.dump(processed_path + '/train.p', ([X_group_train, X_pairs_train], Y_train, times_train))
    utils.dump(processed_path + '/val.p', ([X_group_val, X_pairs_val], Y_val, times_val))

def main():
    expanded = True
    if(expanded): folder = "cocktail_expanded"
    else: folder = "cocktail"

    raw_path = "./datasets/"+folder+"/raw"
    viz_path = "./datasets/"+folder+"/viz"
    clean_path = "./datasets/"+folder+"/clean"
    processed_path = "./datasets/"+folder+"/processed"

    max_people = 20 #max people possible

    #features[0] is space stored for x, y
    #features[1] is space stored for angles related to position
    #features[2] is space stored for all other features
    if(expanded): features = [2, 2, 0]
    else: features = [2, 1, 0]

    X_old, Y_old = load_features(max_people, features, raw_path, viz_path)
    X, Y, times = build_dataset(X_old, Y_old, max_people, features, clean_path)
    save_dataset(X, Y, times, max_people, features, processed_path)

if __name__ == "__main__":
    main()
