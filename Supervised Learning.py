import matplotlib.pyplot as plt
import numpy as np
import statistics

#Loading data to lists
train_file = open("C:/Users/ccane/Desktop/ge461/PROJECT3/train1.txt", "r")
train_list = []
for line in train_file:
    stripped_line = line.strip()
    line_list = stripped_line.split("\t")
    train_list.append(line_list)
print(train_list)

train_list = [list(map(float,i) ) for i in train_list] 

first_train=[]
second_train=[]
for i in range(len(train_list)):
    first_train.append(train_list[i][0])
    second_train.append(train_list[i][1])


test_file = open("C:/Users/ccane/Desktop/ge461/PROJECT3/test1.txt", "r")
test_list = [] 
for line in test_file:
    stripped_line = line.strip()
    line_list = stripped_line.split("\t")
    test_list.append(line_list)

test_list = [list(map(float,i) ) for i in test_list] 
first_test=[]
second_test=[]
for i in range(len(test_list)):
    first_test.append(test_list[i][0])
    second_test.append(test_list[i][1])
listem=[]
#Inputs
liste=[2,4,8,16,32]
for x in liste:
    
    hidden_unitss=x # Used 2, 4, 8, 16, and 32 hidden units
    epochs=10000
    lr=0.002

    weights=[]
    weight1=[]
    weight2=[]
    weight3= []
    for i in range(hidden_unitss):
        weight1.append(np.random.rand())
        weight2.append(np.random.rand())
        weight3.append(1/float(hidden_unitss))
    weights.append(weight1)
    weights.append(weight2)
    weights.append(weight3)
    weight1 = np.array(weight1)
    weight2 = np.array(weight2)
    weight3 = np.array(weight3)


    def activation(fx): # Sigmoid as the activation function to define the hidden units
        return 1 / (1 + (np.exp(- (fx) )))

    def df_sigmoid(fx):
        return fx * (1-fx)
        
    def train(first, second, epoch, lr, weights1, weights2, weights3):
        first=np.array(first)
        second=np.array(second)
        loss_list = []
        for i in range(epoch):
            index = np.random.randint(0,len(second))              
            w1,w2,w_fix = weights1,weights2,weights3
        
            lin = w1 + w2*first[index]
            f_x = activation(lin)
            weighted_sum = np.sum(f_x*w_fix)
            dif = second[index] - weighted_sum     
            weights1 = weights1+lr*dif*w_fix*df_sigmoid(f_x)
            weights2 = weights2+lr*dif*w_fix*first[index]*df_sigmoid(f_x) 
            weights3 = weights3+lr*dif*f_x
            lin = w1 + w2*np.reshape(first,(len(first),1)) 
            f_x = activation(lin)
            loss = np.sum((np.matmul(f_x, w_fix)-second)**2) #sum of the squared errors as your loss function
            loss_list.append(loss)
            if len(loss_list)>5:
                if (loss_list[len(loss_list)-2]+loss_list[len(loss_list)-3]+loss_list[len(loss_list)-4])/4 > loss:
                    print("Break at epoch "+str(i))
                    break
            #print("loss",loss)

        return w1,w2,w_fix

    def predict(first, second, epochs, lr, type,weight1,weight2,weight3):
        weight1,weight2,weight3 = train(first_train, second_train, epochs, lr,weight1,weight2,weight3)
        w1,w2,w_fix = weight1,weight2,weight3
        
        lin = w1 + w2*np.reshape(first,(len(first),1)) 
        fx = activation(lin)
        predictions = np.matmul(fx,w_fix)
        listem.append(predictions)
        plt.legend()
        if type == "train": 
            plt.scatter(first,second, label = 'real data',color="green")
            plt.scatter(first, predictions, label='predictions',color="red")
            loss = np.sum((predictions - second)**2)/len(second)  #sum of the squared errors averaged over training instances
            plt.title("The Actual Outputs For The Given Train Input Points. ") 
            plt.title("Artifical Neural Network on Train Data Loss Average " + str(loss)) 
        else:
            plt.scatter(first,second, label = 'real data',color="green")
            plt.scatter(first, predictions, label='predictions',color="red")
            loss = np.sum((predictions - second)**2)/len(second)  #sum of the squared errors averaged over testing instances
            plt.title("The Actual Outputs For The Given Test Input Points." ) 
            plt.title("Artifical Neural Network on Test Data with Loss Average " + str(loss)) 
        plt.show() 
        print("standard deviations:",statistics.stdev(predictions) )


    predict(first_train,second_train,epochs,lr,"train",weight1,weight2,weight3)
    #predict(first_test,second_test,epochs,lr,"test",weight1,weight2,weight3)


plt.scatter(first_train, listem[0], label='hidden units=2',color="black")
plt.scatter(first_train, listem[1], label='hidden units=4',color="red")
plt.scatter(first_train, listem[2], label='hidden units=8',color="gold")
plt.scatter(first_train, listem[3], label='hidden units=16',color="blue")
plt.scatter(first_train, listem[4], label='hidden units=32',color="green")
plt.legend()
plt.title("Train Set Predictions - Epochs = 10000 and Learning Rate = 0.002")
plt.show()