#centralized constants

#using the expanded dataset with body pose or not
expanded = False

if(expanded): dataset = "cocktail_expanded"
else: dataset = "cocktail"

#paths to various saved files
raw_path = "./datasets/"+dataset+"/raw" #input data
viz_path = "./datasets/"+dataset+"/viz" #groups for visualization
clean_path = "./datasets/"+dataset+"/clean" #all the data transformed and split by i,j
processed_path = "./datasets/"+dataset+"/processed" #split into folds, saved to pickles

max_people = 20 #max people possible (neural network)

#features[0] is space stored for x, y
#features[1] is space stored for angles related to position
#features[2] is space stored for all other features
if(expanded): features = [2, 2, 0] #two theta - one for head, one for body
else: features = [2, 1, 0] #just one theta - for head orientation

#sample model path and new model path
test_model_path = "./models/tests/test0/cocktail/model.h5"
model_path = "./models/"+dataset+"/test1"

#when training model, epochs before updating metrics
skip = 10 #keep 1 for official runs, increase to speed testing
