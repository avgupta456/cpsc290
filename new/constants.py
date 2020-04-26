#centralized constants

expanded = False
if(expanded): dataset = "cocktail_expanded"
else: dataset = "cocktail"

raw_path = "./datasets/"+dataset+"/raw"
viz_path = "./datasets/"+dataset+"/viz"
clean_path = "./datasets/"+dataset+"/clean"
processed_path = "./datasets/"+dataset+"/processed"
processed_path2 = "./datasets/"+dataset+"/processed2"

max_people = 20 #max people possible

#features[0] is space stored for x, y
#features[1] is space stored for angles related to position
#features[2] is space stored for all other features
if(expanded): features = [2, 2, 0]
else: features = [2, 1, 0]

test_model_path = "./models/tests/test0/cocktail/model.h5"
model_path = "./models/"+dataset+"/test1"

skip = 10
