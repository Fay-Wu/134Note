import pandas as pd
import numpy as np
import json
import pickle

with open('review.json') as json_file:      
    data_review = json_file.readlines()
    data_review = list(map(json.loads, data_review)) 
    
len(data_review) # 5,261,669 json files read!

business100 = [business["business_id"] for business in train_business]

# Get list of training reviews from subset of businesses
train_review = [review for review in data_review \
                if review['business_id'] in business100]

pickle.dump(train_review,open('train_review.pkl','wb'))

