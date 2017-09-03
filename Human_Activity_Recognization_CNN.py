
files = ['body_acc_x_train.txt', 'body_acc_y_train.txt','body_acc_z_train.txt',
         'body_gyro_x_train.txt', 'body_gyro_y_train.txt', 'body_gyro_z_train.txt',
         'total_acc_x_train.txt', 'total_acc_y_train.txt', 'total_acc_z_train.txt']
dict_data = {}

for f in files:
    final = []
    fp = open(f)
    temp = fp.readlines()
    fp.close()
    for row in temp:
        x = [float(val) for val in row.strip().split()]
        final.append(x)
    dict_data[f] = final

final_data = []
for i in range(len(final)):
    event = []
    for j in range(len(final[0])):
        column = []
        for f in files:
            column.append(dict_data[f][i][j])
        event.append(column)
    final_data.append(event)

print(len(final_data),len(final_data[0]),len(final_data[0][1]))
# The list final_data has 7352 rows/examples with example having 128 (1*9) matrices to denote reading from each of 9 metrices at a given instant.
