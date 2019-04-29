import numpy as np 
import pandas as pd 
import torch 
import torch.nn as nn 
import torch.nn.parallel 
import torch.optim as optim 
import torch.utils.data
from torch.autograd import Variable

movies = pd.read_csv('ml-1m/movies.dat',#path
sep='::',#seperator default is comma
header=None,
engine='python',
encoding='latin-1'
)

users = pd.read_csv('ml-1m/users.dat',#path
sep='::',#seperator default is comma
header=None,
engine='python',
encoding='latin-1'
)

ratings = pd.read_csv('ml-1m/ratings.dat',
sep='::',
header=None,
engine='python',
encoding='latin-1'
)

#TRAINING DATA

training_set = pd.read_csv('ml-100k/u1.base',
delimiter= '\t'
)
'''
training_set columns contains.
users , columns , ratings, timestamp<useless>
'''

'''
torch variable takes array not dataframes
'''

training_set = np.array(training_set,dtype='int')


test_set = pd.read_csv('ml-100k/u1.test',
delimiter= '\t'
)

test_set = np.array(test_set,dtype='int')

'''
GETTING THE NUMBER OF USERS AND MOVIES
TO CREATE OF MATRIX OF USERS AND MOVIES WHERE 
EACH SHELL REPRESENTS RATING OF USER TO THE MOVIE
'''
total_users = int(max(max(training_set[:,0]),max(test_set[:,0])))
total_movies = int(max(max(training_set[:,1]),max(test_set[:,1])))
'''
CREATING AN MATRIX
USERS IN ROWS 
MOVIES IN COLUMNS
'''

def convert(data):
    new_data=[]
    for id_users in range(1,total_users+1):
        id_movies = data[:,1][data[:,0]==id_users]     #movie ids rated by a user
        id_ratings = data[:,2][data[:,0]==id_users]
        ratings = np.zeros(total_movies)
        ratings[id_movies-1] = id_ratings
        new_data.append(list(ratings))
    return new_data

training_set  = convert(training_set)
test_set = convert(test_set)

training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)


#diffrent full connection b/w the layers - > linear

class SAE(nn.Module):
    def __init__(self,):
        super(SAE,self).__init__()
        self.fc1 = nn.Linear(total_movies,20)
        self.fc2 = nn.Linear(20,10)
        self.fc3 = nn.Linear(10,20)
        self.fc4 = nn.Linear(20,total_movies)
        self.activation = nn.Sigmoid()

    def forward(self,x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        
        x = self.fc4(x) # decoding 
        return x

sae = SAE()
criterion = nn.MSELoss()  #MEAN SQUARED ERROR
optimizer = optim.RMSprop(sae.parameters(),lr=0.01,weight_decay=0.5)

 #TRAINING 

nb_epoch = 400 

for epoch in range(1,nb_epoch+1):
    train_loss = 0 #modified at each epoch
    s = 0.0 # number of users rated atleast 1 movie
    for id_user in range(total_users):
        input = Variable(training_set[id_user]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) > 0: #TO SAVE MEM
            output  = sae.forward(input)
            target.require_grad = False
            output[target==0] = 0 #removing the 0 values
            loss = criterion(output,target)
            #AVERAGE OF ALL THE MOVIES
            mean_corrector = total_movies/float(torch.sum(target.data>0)+1e-10) 
            loss.backward()
            train_loss+= np.sqrt(loss.data.item() * mean_corrector) 
            s+=1.
            optimizer.step()
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))


#TESTING.

test_loss = 0
s = 0 # user that rated atleast one movie.

for id_user in range(total_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user]).unsqueeze(0)
    if torch.sum(target.data > 0) > 0:
        output = sae.forward(input)
        target.require_grad = False
        output[target==0] = 0
        loss = criterion(output, target)
        mean_corrector = total_movies/float(torch.sum(target.data>0) + 1e-10)
        test_loss += np.sqrt(loss.data.item()*mean_corrector)
        s+=1.

print('test loss: '+str(test_loss/s)) 

    
