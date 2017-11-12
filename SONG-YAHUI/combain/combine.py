import csv
import string

class data:
    def __init__(self, id, result):
        self.id = int(id)  # 0
        self.result = int(result)  # 1
    def displayClient(self):
        print (self.id,",", self.result)




data_=[]
data_57648=[]
data_58258=[]
data_58275=[]
data_58337=[]
data_58499=[]
data_58705=[]
data_59121=[]

final_data = []

csvFile = open("data.csv", "r")
reader = csv.reader(csvFile)
for item in reader:
    data_temp = data(item[0],item[1])
    data_.append(data_temp)
    final_data.append(data_temp)
    # data_temp.displayClient()
csvFile.close()


csvFile = open("0.57648.csv", "r")
reader = csv.reader(csvFile)
for item in reader:
    data_temp = data(item[0],item[1])
    data_57648.append(data_temp)
    # data_temp.displayClient()
csvFile.close()

csvFile = open("0.58258.csv", "r")
reader = csv.reader(csvFile)
for item in reader:
    data_temp = data(item[0],item[1])
    data_58258.append(data_temp)
    # data_temp.displayClient()
csvFile.close()

csvFile = open("0.58275.csv", "r")
reader = csv.reader(csvFile)
for item in reader:
    data_temp = data(item[0],item[1])
    data_58275.append(data_temp)
    # data_temp.displayClient()
csvFile.close()


csvFile = open("0.58337.csv", "r")
reader = csv.reader(csvFile)
for item in reader:
    data_temp = data(item[0],item[1])
    data_58337.append(data_temp)
    # data_temp.displayClient()
csvFile.close()

csvFile = open("0.58499.csv", "r")
reader = csv.reader(csvFile)
for item in reader:
    data_temp = data(item[0],item[1])
    data_58499.append(data_temp)
    # data_temp.displayClient()
csvFile.close()


csvFile = open("0.58705.csv", "r")
reader = csv.reader(csvFile)
for item in reader:
    data_temp = data(item[0],item[1])
    data_58705.append(data_temp)
    # data_temp.displayClient()
csvFile.close()


csvFile = open("0.59121.csv", "r")
reader = csv.reader(csvFile)
for item in reader:
    data_temp = data(item[0],item[1])
    data_59121.append(data_temp)
    # data_temp.displayClient()
csvFile.close()


for item in range(len(final_data)):
    temp = int(data_[item].result) + \
        int(data_57648[item].result) + int(data_58258[item].result) + int(data_58275[item].result) + \
        int(data_58337[item].result) +int(data_58499[item].result) +int(data_58705[item].result)+\
        int(data_59121[item].result)
    if temp > 2:
        final_data[item].result = 1
    if temp <= 2:
        final_data[item].result = 0


# write a file;---------------
file_object = open('sampleSubmission.csv', 'w')
file_object.write("id,prediction\n" )
for item in range(len(final_data)):
    # test_client_list[item].displayClient()
    file_object.write("%s," % final_data[item].id )
    file_object.write("%s\n" % final_data[item].result )
file_object.close()
#-----------------------------