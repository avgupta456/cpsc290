import numpy as np
import pickle

def dump(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_matrix(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def load_data(dataset):
    path = "./datasets/" + str(dataset) + "/processed"
    train = load_matrix(path + '/train.p')
    test = load_matrix(path + '/test.p')
    val = load_matrix(path + '/val.p')
    return train, test, val


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
