import pandas as pd

test=pd.read_csv("test.csv")
train=pd.read_csv("train.csv")
#
# body_acc_data=train.iloc[:,0:3]
# label = train['Activity']
#sensor_data=pd.DataFrame(train,columns=['tBodyAcc-mean()-X','tBodyAcc-mean()-Y','tBodyAcc-mean()-Z','Activity'])

body_acc_x_train=pd.read_csv('body_acc_x_train.txt')
body_acc_y_train=pd.read_csv('body_acc_y_train.txt')
body_acc_z_train=pd.read_csv('body_acc_z_train.txt')

x=body_acc_x_train.sum(axis=1)
