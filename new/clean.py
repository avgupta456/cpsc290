import numpy as np
import itertools
import random
import re

def load_features(features_path, groups_path, max_people, features):
    file1 = open(groups_path, "r")
    lines1 = file1.readlines()
    points = len(lines1)

    num_features = features[0] + features[1] + features[2]
    X = np.empty(shape=(points, 1+max_people*(num_features+1)), dtype="U50")
    Y = np.empty(shape=(points, 1+2*max_people), dtype="U50")

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

    file1.close()
    file2.close()

    return X, Y


def transform(pre_transform, i, j, max_people, features):
    pre_features = features[0] + features[1] + features[2]
    post_features = features[0] + 2*features[1] + features[2]
    features_i = pre_transform[0][pre_features*(max_people-2)+1:pre_features*(max_people-1)+1]
    features_j = pre_transform[0][pre_features*(max_people-1)+1:pre_features*max_people+1]

    [xi, yi] = [float(features_i[0]), float(features_i[1])]
    [xj, yj] = [float(features_j[0]), float(features_j[1])]
    [a, b, dx, dy] = [(xi+xj)/2, (yi+yj)/2, (xi-xj)/2, (yi-yj)/2]
    [b0, b1] = [dx/np.sqrt(dx**2+dy**2), dy/np.sqrt(dx**2+dy**2)]

    post_transform = np.empty(shape=(1, 1+max_people*post_features), dtype="U50")
    post_transform[0][0] = pre_transform[0][0]+":"+str(i)+":"+str(j)+":"+"000"

    for k in range(max_people):
        x = float(pre_transform[0][pre_features*k+1])
        y = float(pre_transform[0][pre_features*k+2])

        [x_proj, y_proj] = [b0*(x-a) + b1*(y-b), b1*(x-a) - b0*(y-b)]
        post_transform[0][post_features*k+1:post_features*k+3] = [x_proj, y_proj]

        for m in range(features[1]):
            tx = np.cos(float(pre_transform[0][pre_features*k+3+m]))
            ty = np.sin(float(pre_transform[0][pre_features*k+3+m]))
            [tx_proj, ty_proj] = [b0*tx + b1*ty, b1*tx - b0*ty]
            post_transform[0][post_features*k+3+2*m:post_features*k+3+2*(m+1)] = [tx_proj, ty_proj]

        post_transform[0][post_features*k+3+2*features[1]:post_features*(k+1)] = pre_transform[0][pre_features*k+3+features[1]:pre_features*(k+1)]

    return post_transform

def augment(pre_transform, j, k, max_people, features):
    post_features = features[0] + 2*features[1] + features[2]
    augment_transform = transform(pre_transform, j, k, max_people, features)
    augment_transform[0][0]=augment_transform[0][0][:-1]+"1"

    for i in range(max_people):
        for j in range(1+features[1]):
            augment_transform[0][post_features*i+2*(j+1)] = str(-float(augment_transform[0][post_features*i+2*(j+1)]))

    return augment_transform


def build_dataset(X_old, Y_old, max_people, features):
    pre_features = features[0] + features[1] + features[2]
    post_features = features[0] + 2*features[1] + features[2]

    times = np.empty(shape=(X_old.shape[0], 2), dtype="U50")

    points = 0
    for i in range(len(X_old)):
        people = int((np.where(X_old[i]=="fake")[0][0]-1)/(pre_features+1))
        points += 2*people*(people-1)

    X = np.empty(shape=(points, 1+max_people*post_features), dtype="U50")
    Y = np.empty(shape=(points, 2), dtype="U50")

    pre_transform = np.empty(shape=(1, 1+max_people*pre_features), dtype="U50")

    pos = 0
    for i in range(len(X_old)):
        people = int((np.where(X_old[i]=="fake")[0][0]-1)/(pre_features+1))
        combinations = [[p1, p2] for p1 in range(people) for p2 in range(people) if p1!=p2]
        for func in [transform, augment]:
            for j, k in combinations:
                
                pre_transform[0][0] = X_old[i][0]
                count = 0

                for m in range(people):
                    if(m!=j and m!=k):
                        pre_transform[0][pre_features*count+1:pre_features*(count+1)+1] = X_old[i][(pre_features+1)*m+2:(pre_features+1)*(m+1)+1]
                        count += 1

                for m in range(people, max_people):
                    rand_p = random.randint(0, people-3) #includes final number
                    pre_transform[0][pre_features*count+1:pre_features*(count+1)+1] = pre_transform[0][pre_features*rand_p+1:pre_features*(rand_p+1)+1]
                    count += 1

                pre_transform[0][pre_features*count+1:pre_features*(count+1)+1] = X_old[i][(pre_features+1)*j+2:(pre_features+1)*(j+1)+1]
                pre_transform[0][pre_features*(count+1)+1:pre_features*(count+2)+1] = X_old[i][(pre_features+1)*k+2:(pre_features+1)*(k+1)+1]

                post_transform = func(pre_transform, j, k, max_people, features)

                affinity = 1 if Y_old[i][2*(j+1)]==Y_old[i][2*(k+1)] else 0

                X[pos] = post_transform
                Y[pos] = [X[pos][0], affinity]
                pos += 1

    return X, Y

features_path = "./datasets/cocktail/raw/features.txt"
groups_path = "./datasets/cocktail/raw/groups.txt"

max_people = 20 #max people possible

f_pos_xy = 2 #space stored for x, y
f_pos_angle = 1 #space stored for angles related to position
f_normal = 0 #space stored for all other features
features = [f_pos_xy, f_pos_angle, f_normal]
null = [0] * (f_pos_xy+f_pos_angle+f_normal)

X_old, Y_old = load_features(features_path, groups_path, max_people, features)
X, Y = build_dataset(X_old, Y_old, max_people, features)
print(X)
print(Y)

print(X.shape)
print(Y.shape)
