import numpy as np
import random
import re
import os

from helper import utils
import constants

#input: raw features from dataset
#action: cross references group, position data to extract frames where ground truth exists
#output: X array of original data, Y array of (ID, group_num) pairs where ground truth exists
def load_features(max_people, features, raw_path, viz_path):
    file1 = open(raw_path + "/groups.txt", "r")
    lines1 = file1.readlines()
    points = len(lines1)

    #creates output matrices
    num_features = features[0] + features[1] + features[2]
    X = np.empty(shape=(points, 1+max_people*(num_features+1)), dtype="U50")
    Y = np.empty(shape=(points, 1+2*max_people), dtype="U50")

    file2 = open(raw_path + "/features.txt")
    lines2 = file2.readlines()

    pos = 0
    time = float(lines1[pos].split()[0])

    #iterating over lines in features.txt
    for i in range(len(lines2)):
        line = lines2[i].split() #checks if within time
        if abs(float(line[0])-time)<0.05 and pos<points:
            X[pos][0] = time #stores time
            people = int((len(line)-1)/(num_features+1))

            #transfer input features
            for j in range(people):
                X[pos][(num_features+1)*j+1:(num_features+1)*(j+1)+1] = line[(num_features+1)*j+1:(num_features+1)*(j+1)+1]

            #adds fake people as necessary
            for j in range(people, max_people):
                X[pos][(num_features+1)*j+1] = "fake"
                X[pos][(num_features+1)*j+2:(num_features+1)*(j+1)+1] = [0]*num_features

            Y[pos][0] = time
            #gets list of groups
            groups_arr = re.split(" < | > < ", lines1[pos].strip())
            ids_arr = {} #makes dictionary
            #iterating over list of groups
            for j in range(1,len(groups_arr)):
                ids = groups_arr[j].split() #gets ids in the group
                if(j==len(groups_arr)-1): ids = ids[:-1] #if last group, ignore
                for id in ids: ids_arr[id] = j #adds id to dictionary

            #iterating over real people, adds name and grouping
            for j in range(people):
                Y[pos][2*j+1] = X[pos][(num_features+1)*j+1]
                Y[pos][2*j+2] = ids_arr[Y[pos][2*j+1]]

            #iterating over fake people, adds name and grouping
            for j in range(people, max_people):
                Y[pos][2*j+1] = "fake"
                Y[pos][2*j+2] = -1

            #updates position in group file, gets next time to search for
            pos = min(pos+1, points-1) #just avoids out of bounds
            time = float(lines1[pos].split()[0]) #gets next time

    file1.close()
    file2.close()

    #saves to viz folder
    np.savetxt(viz_path + "/X.txt", X, fmt='%s')
    np.savetxt(viz_path + "/Y.txt", Y, fmt='%s')

    return X, Y

#inputs: the input features in X, and name/group for each person in Y, for a single frame with truth annotation
#action: converts angles into components, handles fakes, splits by i,j, sends through coordinate transform
#output: processed frames in new coordinate system (multiple frames for a single input due to combinations of (i,j), func choice
def process_frame(X_frame, Y_frame, max_people, features, func=utils.transform):
    #func is either transform or augment, centers the coordinate system around two people
    pre_features = features[0] + features[1] + features[2] #calculates before features count
    post_features = features[0] + 2*features[1] + features[2] #after feature count

    #number of people, number of new rows created
    people = int((np.where(X_frame=="fake")[0][0]-1)/(pre_features+1))
    points = people*(people-1) # (don't divide by 2 because both transform and augment)

    #new arrays storing transformation
    pre_transform = np.empty(shape=(1, 1+max_people*pre_features), dtype="U50")
    X = np.empty(shape=(points, 1+max_people*post_features), dtype="U50")
    Y = np.empty(shape=(points, 2), dtype="U50")

    #everything in people^2 ignoring duplicates
    combinations = [[p1, p2] for p1 in range(people) for p2 in range(people) if p1!=p2]
    pos = 0

    #iterating over combinations
    for j, k in combinations:
        pre_transform[0][0] = X_frame[0]
        count = 0

        for m in range(people):
            if(m!=j and m!=k):
                #for all people not i or j, just copy over, but don't save space for i and j (append i and j to the end isntead)
                pre_transform[0][pre_features*count+1:pre_features*(count+1)+1] = X_frame[(pre_features+1)*m+2:(pre_features+1)*(m+1)+1]
                count += 1 #to keep track when filling in i and j's locations

        #iterating over fakes
        for m in range(people, max_people):
            rand_p = random.randint(0, people-3) #random number among the standard people, randint includes final number
            pre_transform[0][pre_features*count+1:pre_features*(count+1)+1] = pre_transform[0][pre_features*rand_p+1:pre_features*(rand_p+1)+1]
            count += 1 #essentially placing a duplicate person from the crowd to avoid issues with max pooling and symmetric operations in the model

        #adds i and j to the end
        pre_transform[0][pre_features*count+1:pre_features*(count+1)+1] = X_frame[(pre_features+1)*j+2:(pre_features+1)*(j+1)+1]
        pre_transform[0][pre_features*(count+1)+1:pre_features*(count+2)+1] = X_frame[(pre_features+1)*k+2:(pre_features+1)*(k+1)+1]

        #applies the coordinate transform to the newly constructed matrices
        post_transform = func(pre_transform, j, k, max_people, features)
        affinity = 1 if Y_frame[2*(j+1)]==Y_frame[2*(k+1)] else 0 #stores the affinity between i and j

        X[pos] = post_transform #stores in the larger array
        Y[pos] = [X[pos][0], affinity] #stores in larger array
        pos += 1

    #returns transformed coordinates
    return X, Y, pos

#inputs: X array of original data, Y array of groupings, for all frames where ground truth exists
#action: uses process frame to process each individual frame, stores in a larger matrix
#output: processed data for all the frames, using both transforms, for each i,j pairing
def build_dataset(X_old, Y_old, max_people, features, clean_path):
    pre_features = features[0] + features[1] + features[2]
    post_features = features[0] + 2*features[1] + features[2]

    #stores timechanges to avoid splitting samples when passing into future methdos
    times = np.empty(shape=(2*X_old.shape[0], 1), dtype="U50")
    times_pos = 0

    points = 0
    for i in range(len(X_old)):
        #counts number of points needed when finished
        people = int((np.where(X_old[i]=="fake")[0][0]-1)/(pre_features+1))
        points += 2*people*(people-1)

    #creates output matrices with correct number of points, using floats now
    X = np.empty(shape=(points, 1+max_people*post_features), dtype="U50")
    Y = np.empty(shape=(points, 2), dtype="U50")

    pos = 0
    #iterating over old frames
    for i in range(len(X_old)):
        #process using the transform function
        X_temp, Y_temp, points_temp = process_frame(X_old[i], Y_old[i], max_people, features, utils.transform)
        X[pos:pos+points_temp], Y[pos:pos+points_temp], times[times_pos] = X_temp, Y_temp, pos #stores
        pos, times_pos = pos + points_temp, times_pos + 1 #updates iterators

        #repeats with the augment function (inverting y-coordinates)
        X_temp, Y_temp, points_temp = process_frame(X_old[i], Y_old[i], max_people, features, utils.augment)
        X[pos:pos+points_temp], Y[pos:pos+points_temp], times[times_pos] = X_temp, Y_temp, pos
        pos, times_pos = pos + points_temp, times_pos + 1

    #saves data to clean path
    np.savetxt(clean_path + "/coordinates.txt", X, fmt='%s')
    np.savetxt(clean_path + "/affinities.txt", Y, fmt='%s')
    np.savetxt(clean_path + "/timechanges.txt", times, fmt='%s')

    #returns for future use
    return X, Y, times

#input: processed data, percentage values to split at
#actions: splits the data while preserving full frames, updates timechanges accordingly
#output: list of smaller arrays following desired split ratios, each contianing full frames only
def split(input_data, splits):
    count = len(splits)
    points = input_data[0].shape[0]
    times = input_data[3]

    cutoffs = []
    data = []
    for i in range(count):
        if(i==0): prev_index, prev_time_index = 0, 0 #handles first cut
        else: prev_index, prev_time_index = cutoffs[-1] #handles all subsequent

        index = int(splits[i]*points) #roughly the cutoff
        while(index not in times): index-=1 #move backwards until at a timechange
        time_index = np.where(times==index)[0][0] #the cutoff in the timechange list

        if(i==count-1): index, time_index = points, times.shape[0] #handles last cut

        #slices input
        X_group_fold = input_data[0][prev_index:index]
        X_pair_fold = input_data[1][prev_index:index]
        Y_fold = input_data[2][prev_index:index]

        #calculates offset and builds matrix with that value
        offset = np.ones(shape=(time_index-prev_time_index, 1))*prev_index

         #subtracts offset so timechanges starts at 0
        times_fold = input_data[3][prev_time_index:time_index] - offset

        cutoffs.append([index, time_index]) #appends for next iterations starting
        data.append([X_group_fold, X_pair_fold, Y_fold, times_fold]) #appends data to final output

    return data

#input: the processed data split into multiple cuts
#action: pulls one cut aside one by one, and bundles the rest together (treating timechanges carefully)
#output: folded datasets where each is missing exactly one of the split data input
def create_folds(split_data):
    num_folds = len(split_data)
    folds, out_folds = [], []

    #iterating over each cut to exclude
    for i in range(num_folds):
        started = False
        #each cut must go in or out
        for j in range(num_folds):
            if(i==j):
                out_folds.append(split_data[j]) #oddo ne out

            elif(not started):
                X_group_fold, X_pairs_fold, Y_fold, times_fold = split_data[j]
                started = True #if first, start with its values (since can't append)

            else:
                offset = np.ones(shape=split_data[j][3].shape)*X_group_fold.shape[0] #calculate time offset
                times_fold = np.append(times_fold, split_data[j][3]+offset, axis=0) #update times and append
                X_group_fold = np.append(X_group_fold, split_data[j][0], axis=0) #append all data
                X_pairs_fold = np.append(X_pairs_fold, split_data[j][1], axis=0)
                Y_fold = np.append(Y_fold, split_data[j][2], axis=0)

        #having iterated, append finished fold to list
        folds.append([X_group_fold, X_pairs_fold, Y_fold, times_fold])

    #returns list of folds and exclusions
    return folds, out_folds

#input: processed frames for X and Y (all frames, transforms, (i,j) pairings)
#action: uses split() and create_folds() to separate into train, val, test
#output: for each fold, splits into train, val, test (handling timechanges), saves to file
def save_dataset(X_old, Y_old, times_old, max_people, features, processed_path):
    post_features = features[0] + 2*features[1] + features[2]
    points = X_old.shape[0] #constants to save retyping

    #new matrices (X split into group and dyad), Y to be int, time now in numpy
    X_group = np.zeros(shape=(points, 1, max_people-2, post_features))
    X_pairs = np.zeros(shape=(points, 1, 2, post_features))
    Y = np.zeros(shape=(points, 1), dtype=np.int8)
    times = np.zeros(shape=(len(times_old), 1), dtype=np.int32)

    #iterating over all inputs
    for i in range(points):
        X_group[i][0] = np.reshape(X_old[i][1:-2*post_features], newshape=(max_people-2, post_features)) #fills X and Y
        X_pairs[i][0] = np.reshape(X_old[i][-2*post_features:], newshape=(2, post_features))
        Y[i][0] = int(Y_old[i][1])

    for i in range(times.shape[0]): times[i]=float(times_old[i][0]) #fills timechanges

    data = split([X_group, X_pairs, Y, times], [0.2, 0.4, 0.6, 0.8, 1.0]) #splits into five equal chunks
    folds, out_folds = create_folds(data) #creates folds using the splits

    #for each fold
    for i in range(len(folds)):
        train, val = split(folds[i], [0.75, 1.0]) #splits the fold into train and val
        test = out_folds[i] #the out fold is used for test

        #bundles into one list
        train = [[train[0], train[1]], train[2], train[3]]
        test = [[test[0], test[1]], test[2], test[3]]
        val = [[val[0], val[1]], val[2], val[3]]

        #pickles it away
        dir = processed_path + "/fold"+str(i)
        if not os.path.isdir(dir): os.makedirs(dir)
        utils.dump(dir+"/train.p", train)
        utils.dump(dir+"/test.p", test)
        utils.dump(dir+"/val.p", val)

#main function running all the above
def main():
    #loads constants
    max_people = constants.max_people
    features = constants.features

    #loads paths
    raw_path = constants.raw_path
    viz_path = constants.viz_path
    clean_path = constants.clean_path
    processed_path = constants.processed_path

    #runs methods
    X_old, Y_old = load_features(max_people, features, raw_path, viz_path)
    X, Y, times = build_dataset(X_old, Y_old, max_people, features, clean_path)
    save_dataset(X, Y, times, max_people, features, processed_path)

if __name__ == "__main__":
    main()
