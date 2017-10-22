import csv
import string

class Item_client:
    def __init__(self, id, Output):
        self.id = id
        self.Output = Output
        self.grade = 0

    def displayClient(self):
        print(self.id, self.Output)


train_list = [];
submission_list =[];
count_yes = 0;
same = 0;
same_yes = 0;

# read a train file;----------
csvFile = open("train.csv", "r")
reader = csv.reader(csvFile)
for item in reader:
    if item[21] == "yes":
        client_temp = Item_client(item[0],1)
    if item[21] == "no":
        client_temp = Item_client(item[0],0)
    train_list.append(client_temp)
    # client_temp.displayClient()
csvFile.close()


csvFile = open("sampleSubmission1.csv", "r")
reader = csv.reader(csvFile)

for item in reader:
    client_temp = Item_client(item[0],item[1])
    client_temp.grade = item[2]
    submission_list.append(client_temp)
    # client_temp.displayClient()
csvFile.close()


for i in range(0,len(train_list)):
    if train_list[i].Output ==1:
        count_yes = count_yes +1
    if train_list[i].Output == (int(submission_list[i].Output)):
        same = same + 1;
        if train_list[i].Output ==1:
            same_yes = same_yes +1;
    if train_list[i].Output != (int(submission_list[i].Output)) and (int(submission_list[i].Output)) == 1 :
        print(submission_list[i].id,submission_list[i].grade )
    # print(train_list[i].id,train_list[i].Output,submission_list[i].Output,(train_list[i].Output == (int(submission_list[i].Output))))

print(same/len(train_list), same_yes/count_yes)
# print(count_yes/len(train_list)*48)