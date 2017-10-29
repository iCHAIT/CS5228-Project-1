import csv
import string

class data:
    def __init__(self, id, result):
        self.id = int(id)  # 0
        self.result = int(result)  # 1
    def displayClient(self):
        print (self.id,",", self.result)


algorithem = ['NB', 'KNN', 'LR', 'RF', 'DT', 'SVM','GBDT']

NB_data = []
KNN_data = []
LR_data = []
RF_data = []
DT_data = []
SVM_data = []
GBDT_data = []
final_data = []

csvFile = open("sampleSubmissionNB.csv", "r")
reader = csv.reader(csvFile)
for item in reader:
    data_temp = data(item[0],item[1])
    NB_data.append(data_temp)
    final_data.append(data_temp)
    # data_temp.displayClient()
csvFile.close()


csvFile = open("sampleSubmissionKNN.csv", "r")
reader = csv.reader(csvFile)
for item in reader:
    data_temp = data(item[0],item[1])
    KNN_data.append(data_temp)
    # data_temp.displayClient()
csvFile.close()


csvFile = open("sampleSubmissionLR.csv", "r")
reader = csv.reader(csvFile)
for item in reader:
    data_temp = data(item[0],item[1])
    LR_data.append(data_temp)
    # data_temp.displayClient()
csvFile.close()


csvFile = open("sampleSubmissionRF.csv", "r")
reader = csv.reader(csvFile)
for item in reader:
    data_temp = data(item[0],item[1])
    RF_data.append(data_temp)
    # data_temp.displayClient()
csvFile.close()


csvFile = open("sampleSubmissionDT.csv", "r")
reader = csv.reader(csvFile)
for item in reader:
    data_temp = data(item[0],item[1])
    DT_data.append(data_temp)
    # data_temp.displayClient()
csvFile.close()


csvFile = open("sampleSubmissionSVM.csv", "r")
reader = csv.reader(csvFile)
for item in reader:
    data_temp = data(item[0],item[1])
    SVM_data.append(data_temp)
    # data_temp.displayClient()
csvFile.close()


csvFile = open("sampleSubmissionGBDT.csv", "r")
reader = csv.reader(csvFile)
for item in reader:
    data_temp = data(item[0],item[1])
    GBDT_data.append(data_temp)
    # data_temp.displayClient()
csvFile.close()


for item in range(len(final_data)):
    temp = int(NB_data[item].result) + int(KNN_data[item].result) + int(LR_data[item].result) + \
        int(RF_data[item].result) + int(DT_data[item].result) + int(SVM_data[item].result) + \
        int(GBDT_data[item].result)
    if temp > 1:
        final_data[item].result = 1
    if temp <= 1:
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