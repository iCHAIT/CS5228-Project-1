import csv


class Client: # a class to of client, used to save all the data.

    def __init__(self, id, age, job, marital, education, default, housing, loan, contact, month, day_of_week, duration, campaign, pdays, previous, poutcome, emp_var_rate, cons_price_idx, cons_conf_idx, euribor3m, nr_employed, Output):
        self.id = id                                        # 0
        self.age = age                                 # 1
        self.job = job                             # 2
        self.marital = marital                # 3
        self.education = education           # 4
        self.default = default                 # 5
        self.housing = housing                   # 6
        self.loan = loan                         # 7
        self.contact = contact                 # 8
        self.month = month                       # 9
        self.day_of_week = day_of_week            # 10
        self.duration = duration                        # 11
        self.campaign = campaign                        # 12
        self.pdays = pdays                             # 13
        self.previous = previous                       # 14
        self.poutcome = poutcome              # 15
        self.emp_var_rate = emp_var_rate              # 16
        self.cons_price_idx = cons_price_idx         # 17
        self.cons_conf_idx = cons_conf_idx            # 18
        self.euribor3m = euribor3m                  # 19
        self.nr_employed = nr_employed               # 20
        self.Output = Output                    # 21

    def displayClient(self):
        print (self.id, self.age, self.job, self.marital, self.education, self.default, self.housing, self.loan, self.contact, self.month, self.day_of_week, self.duration, self.campaign, self.pdays, self.previous, self.poutcome, self.emp_var_rate, self.cons_price_idx, self.cons_conf_idx, self.euribor3m, self.nr_employed, self.Output)

class Map_song:
    def __init__(self,id,result):
        self.id = id                                        # 0
        self.result = int(result)                                 # 1
    def display(self):
        print(self.id,self.result)


def write_file_test(name,list):
    # write a file;---------------
    file_object = open(name, 'w')

    for item in range(len(list)):
        file_object.write("%s," % list[item].id)
        file_object.write("%s," % list[item].age)
        file_object.write("%s," % list[item].job)
        file_object.write("%s," % list[item].marital)
        file_object.write("%s," % list[item].education)
        file_object.write("%s," % list[item].default )
        file_object.write("%s," % list[item].housing)
        file_object.write("%s," % list[item].loan)
        file_object.write("%s," % list[item].contact)
        file_object.write("%s," % list[item].month)
        file_object.write("%s," % list[item].day_of_week)
        file_object.write("%s," % list[item].duration)
        file_object.write("%s," % list[item].campaign)
        file_object.write("%s," % list[item].pdays)
        file_object.write("%s," % list[item].previous)
        file_object.write("%s," % list[item].poutcome )
        file_object.write("%s," % list[item].emp_var_rate)
        file_object.write("%s," % list[item].cons_price_idx)
        file_object.write("%s," % list[item].cons_conf_idx)
        file_object.write("%s," % list[item].euribor3m)
        file_object.write("%s," % list[item].nr_employed)
        file_object.write("%s\n" % list[item].Output)

    file_object.close()
    # -----------------------------

ground_truth=[]

csvFile = open("true.csv", "r")
reader = csv.reader(csvFile)
for item in reader:
    map_temp = Map_song(item[0],item[1])
    ground_truth.append(map_temp)
    # map_temp.display()
csvFile.close()


flie_list=[]

csvFile = open("originTest.csv", "r")
reader = csv.reader(csvFile)
for item in reader:
    client_temp = Client(item[0],item[1],item[2],item[3],item[4],item[5],item[6],item[7],item[8],item[9],item[10],item[11],item[12],item[13],item[14],item[15],item[16],item[17],item[18],item[19],item[20],"yes")

    flie_list.append(client_temp)
    # client_temp.displayClient()
csvFile.close()


for item in range(0,len(flie_list)):
    if ground_truth[item].result == 1:
        flie_list[item].Output = "yes"
    if ground_truth[item].result == 0:
        flie_list[item].Output = "no"
    flie_list[item].displayClient()


# write a file;---------------
file_object = open("originTest_truth.csv", 'w')
'''
file_object.write("id,"
                      "age,"
                      "job,"
                      "marital,"
                      "education,"
                      "default,"
                      "housing,"
                      "loan,"
                      "contact,"
                      "month,"
                      "day_of_week,"
                      "duration,"
                      "campaign,"
                      "pdays,"
                      "previous,"
                      "poutcome,"
                      "emp.var.rate,"
                      "cons.price.idx,"
                      "cons.conf.idx,"
                      "euribor3m,"
                      "nr.employed,"
                  "y\n")
'''
for item in range(len(flie_list)):
    file_object.write("%s," % flie_list[item].id)

    file_object.write("%s," % flie_list[item].age)
    file_object.write("%s," % flie_list[item].job)
    file_object.write("%s," % flie_list[item].marital)
    file_object.write("%s," % flie_list[item].education)
    file_object.write("%s," % flie_list[item].default)
    file_object.write("%s," % flie_list[item].housing)
    file_object.write("%s," % flie_list[item].loan)
    file_object.write("%s," % flie_list[item].contact)
    file_object.write("%s," % flie_list[item].month)
    file_object.write("%s," % flie_list[item].day_of_week)
    file_object.write("%s," % flie_list[item].duration)
    file_object.write("%s," % flie_list[item].campaign)
    file_object.write("%s," % flie_list[item].pdays)
    file_object.write("%s," % flie_list[item].previous)
    file_object.write("%s," % flie_list[item].poutcome)
    file_object.write("%s," % flie_list[item].emp_var_rate)
    file_object.write("%s," % flie_list[item].cons_price_idx)
    file_object.write("%s," % flie_list[item].cons_conf_idx)
    file_object.write("%s," % flie_list[item].euribor3m)
    file_object.write("%s," % flie_list[item].nr_employed)
    file_object.write("%s\n" % flie_list[item].Output)

file_object.close()
    # -----------------------------