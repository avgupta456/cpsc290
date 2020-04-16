import numpy as np
import random
import re
import os

from helper import utils
import constants

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

def split(input_data, splits):
    count = len(splits)
    points = input_data[0].shape[0]
    times = input_data[3]

    cutoffs = []
    data = []
    for i in range(count):
        if(i==0): prev_index, prev_time_index = 0, 0
        else: prev_index, prev_time_index = cutoffs[-1]

        index = int(splits[i]*points)
        while(index not in times): index-=1
        time_index = np.where(times==index)[0][0]

        if(i==count-1): index, time_index = points, times.shape[0]

        X_group_fold = input_data[0][prev_index:index]
        X_pair_fold = input_data[1][prev_index:index]
        Y_fold = input_data[2][prev_index:index]

        offset = np.ones(shape=(time_index-prev_time_index, 1))*prev_index
        times_fold = input_data[3][prev_time_index:time_index] - offset

        cutoffs.append([index, time_index])
        data.append([X_group_fold, X_pair_fold, Y_fold, times_fold])

    return data

def create_folds(split_data):
    num_folds = len(split_data)
    folds, out_folds = [], []

    for i in range(num_folds):
        started = False
        for j in range(num_folds):
            if(i==j):
                out_folds.append(split_data[j])

            elif(not started):
                X_group_fold, X_pairs_fold, Y_fold, times_fold = split_data[j]
                started = True

            else:
                offset = np.ones(shape=split_data[j][3].shape)*X_group_fold.shape[0]
                times_fold = np.append(times_fold, split_data[j][3]+offset, axis=0)
                X_group_fold = np.append(X_group_fold, split_data[j][0], axis=0)
                X_pairs_fold = np.append(X_pairs_fold, split_data[j][1], axis=0)
                Y_fold = np.append(Y_fold, split_data[j][2], axis=0)

        folds.append([X_group_fold, X_pairs_fold, Y_fold, times_fold])

    return folds, out_folds

def save_dataset(X_old, Y_old, times_old, max_people, features, processed_path):
    post_features = features[0] + 2*features[1] + features[2]
    points = X_old.shape[0]

    X_group = np.zeros(shape=(points, 1, max_people-2, post_features))
    X_pairs = np.zeros(shape=(points, 1, 2, post_features))
    Y = np.zeros(shape=(points, 1), dtype=np.int8)
    times = np.zeros(shape=(len(times_old), 1), dtype=np.int32)
    for i in range(times.shape[0]): times[i]=float(times_old[i][0])

    for i in range(points):
        X_group[i][0] = np.reshape(X_old[i][1:-2*post_features], newshape=(max_people-2, post_features))
        X_pairs[i][0] = np.reshape(X_old[i][-2*post_features:], newshape=(2, post_features))
        Y[i][0] = int(Y_old[i][1])

    data = split([X_group, X_pairs, Y, times], [0.2, 0.4, 0.6, 0.8, 1.0])
    folds, out_folds = create_folds(data)

    for i in range(len(folds)):
        train, val = split(folds[i], [0.75, 1.0])
        test = out_folds[i]

        train = [[train[0], train[1]], train[2], train[3]]
        test = [[test[0], test[1]], test[2], test[3]]
        val = [[val[0], val[1]], val[2], val[3]]

        dir = processed_path + "/fold"+str(i)
        if not os.path.isdir(dir): os.makedirs(dir)
        utils.dump(dir+"/train.p", train)
        utils.dump(dir+"/test.p", test)
        utils.dump(dir+"/val.p", val)

def main():
    max_people = constants.max_people
    features = constants.features

    raw_path = constants.raw_path
    viz_path = constants.viz_path
    clean_path = constants.clean_path
    processed_path = constants.processed_path

    X_old, Y_old = load_features(max_people, features, raw_path, viz_path)
    X, Y, times = build_dataset(X_old, Y_old, max_people, features, clean_path)
    save_dataset(X, Y, times, max_people, features, processed_path)

if __name__ == "__main__":
    main()
