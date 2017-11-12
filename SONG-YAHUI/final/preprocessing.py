import csv
import random

relax = 2

def chackJOB(job):
    temp = job
    if temp == "unknown":
        de = random.randint(0, 100)
        if de <= 25 : temp = "admin."
        if de > 25 and de <=28: temp= "self-employed"
        if de > 28 and de <=45: temp= "technician"
        if de > 45 and de <=52: temp= "management"
        if de > 52 and de <=62: temp= "services"
        if de > 62 and de <=64: temp= "student"
        if de > 64 and de <=66: temp= "unemployed"
        if de > 66 and de <=69: temp= "housemaid"
        if de > 69 and de <=92: temp= "blue-collar"
        if de > 92 and de <=96: temp= "entrepreneur"
        if de > 96: temp= "retired"

    if temp == "admin.": return 0
    if temp == "self-employed": return 1
    if temp == "technician": return 2
    if temp == "management": return 3
    if temp == "services": return 4
    if temp == "student": return 5
    if temp == "unemployed": return 6
    if temp == "housemaid": return 7
    if temp == "blue-collar": return 8
    if temp == "entrepreneur": return 9
    if temp == "retired": return 10

def chackEDUCATION(education):
    temp = education
    if temp == "unknown":
        de = random.randint(0, 100)
        if de <= 24: temp = "high.school"
        if de > 24 and de <= 55: temp = "university.degree"
        if de > 55 and de <= 61: temp = "basic.6y"
        if de > 61 and de <= 72: temp = "basic.4y"
        if de > 72 and de <= 87: temp = "basic.9y"
        if de > 87 : temp = "professional.course"

    if temp  == "high.school": return 0
    if temp == "university.degree": return 1
    if temp == "basic.6y": return 2
    if temp == "basic.4y": return 3
    if temp == "basic.9y": return 4
    if temp == "professional.course": return 5
    if temp == "illiterate": return 6

def chackMARITAL(marital):
    temp = marital
    if temp  == "unknown":
        de = random.randint(0, 100)
        if de <= 28: temp = "single"
        if de > 28 and de <= 39: temp = "divorced"
        if de > 39: temp = "married"
    if temp == "single": return 0
    if temp == "divorced": return 1
    if temp == "married": return 2


def chackHOUSE(housing):
    temp = housing
    if temp == "unknown":
        de = random.randint(0, 100)
        if de < 46: temp = "no"
        if de >= 46 : temp = "yes"
    if temp  == "no": return 0
    if temp  == "yes": return 1

def chackLOAN(loan):
    temp = loan
    if temp == "unknown":
        de = random.randint(0, 100)
        if de < 85: temp= "no"
        if de >= 85: temp= "yes"
    if temp  == "no": return 0
    if temp  == "yes": return 1

def chackDEFAULT(default):
    if default == "no": return 0
    if default == "yes": return 1
    if default == "unknown": return 2

def chackOUTPUT(Output):
    if Output == "no": return 0
    if Output == "yes": return 1


def chackMOUTH(month):
    if month == "may": return 0
    if month == "nov": return 1
    if month == "apr": return 2
    if month == "aug": return 3
    if month == "sep": return 4
    if month == "jun": return 5
    if month == "oct": return 6
    if month == "jul": return 7
    if month == "dec": return 8
    if month == "mar": return 9

def chackWEEK(day_of_week):
    if day_of_week == "mon": return 1
    if day_of_week == "tue": return 2
    if day_of_week == "wed": return 3
    if day_of_week == "thu": return 4
    if day_of_week == "fri": return 5

def chackCONTACK(contact):
    if contact == "cellular": return 0 # 1.31;
    if contact == "telephone": return 1 # 0.46;

def chackPOUTCOME(poutcome):
    if poutcome == "failure": return 0
    if poutcome == "nonexistent": return 1 # 0.77;
    if poutcome == "success": return  2 #5.84;

################Numeric#####################################


def chackPdays(num):
    return num
    if num == 999: return 1 # 0.81
    return num / 30

def chackDURA(num):
    return num # (num/420)
    if num <= 50: return 0
    if num > 50 and num <= 100: return 1
    if num > 100 and num <= 150: return 2
    if num > 150 and num <= 200: return 3
    if num > 200 and num <= 250: return 4
    if num > 250 and num <= 300: return 5
    if num > 300 and num <= 350: return 6
    if num > 350 and num <= 400: return 7
    if num > 400 and num <= 450: return 8
    if num > 450 and num <= 500: return 9
    if num > 500 and num <= 550: return 10
    if num > 550 and num <= 600: return 11
    if num > 600 and num <= 650: return 12
    if num > 650 and num <= 700: return 13
    if num > 700: return 14



def chackCAMP(num):
    return num # num/40
    if num <= 10: return 0 # 1.01
    if num > 10 and num <= 20: return 0.25 # 0.37
    if num > 20 and num <= 30: return 0.5 # 0.09
    if num > 30 and num <= 40: return 0.75 # 0
    if num > 40: return 1 # 0

def checkExtra(pdays,poutcome):
    if poutcome == "nonexistent": return 0
    if pdays == 999 and poutcome == "failure": return 1  # 0.81
    return 2

train_list=[]
test_list=[]
test_true_list=[]


class Client: # a class to of client, used to save all the data.

    def __init__(self, id, age, job, marital, education, default, housing, loan, contact, month, day_of_week, duration, campaign, pdays, previous, poutcome, emp_var_rate, cons_price_idx, cons_conf_idx, euribor3m, nr_employed, Output):
        self.id = id                                        # 0
    ##########################

        self.job = chackJOB(job)  # 2
        self.marital = chackMARITAL(marital)  # 3
        self.education = chackEDUCATION(education)  # 4
        self.default = chackDEFAULT(default)  # 5
        self.housing = chackHOUSE(housing)  # 6
        self.loan = chackLOAN(loan)  # 7
        self.contact = chackCONTACK(contact)  # 8
        self.month = chackMOUTH(month)  # 9
        self.day_of_week = chackWEEK(day_of_week)  # 10
        self.poutcome = chackPOUTCOME(poutcome)              # 15

    ############################
        self.age = (int(age) - 17) / 81 *relax # 1
        self.duration = int(duration) # chackDURA(int(duration)) / 4199 *relax # 11
        self.campaign = chackCAMP(int(campaign)) / 56 *relax # 12
        self.pdays = int(pdays) # 13
        self.previous = int(previous) / 7 *relax # 14
        self.emp_var_rate = (float(emp_var_rate) + 3.4) / 4.8 *relax # 16
        self.cons_price_idx = (float(cons_price_idx) - 92.201) / 2.6 *relax # 17
        self.cons_conf_idx = (float(cons_conf_idx) + 50.8) / 24 *relax # 18
        self.euribor3m = (float(euribor3m) - 0.634) / 4.411 *relax # 19
        self.nr_employed = (float(nr_employed) - 4963) / 265 *relax # 20


        self.extra_feature = checkExtra(pdays,poutcome)

        self.Output = chackOUTPUT(Output)                    # 21

    def displayClient(self):
        print (self.id, self.age, self.job, self.marital, self.education, self.default, self.housing, self.loan, self.contact, self.month, self.day_of_week, self.duration, self.campaign, self.pdays, self.previous, self.poutcome, self.emp_var_rate, self.cons_price_idx, self.cons_conf_idx, self.euribor3m, self.nr_employed, self.Output)


# read a file;----------
def read_file(name,list):
    csvFile = open(name, "r")
    reader = csv.reader(csvFile)
    for item in reader:
        client_temp = Client(item[0],item[1],item[2],item[3],item[4],item[5],item[6],item[7],item[8],item[9],item[10],item[11],item[12],item[13],item[14],item[15],item[16],item[17],item[18],item[19],item[20],item[21])
        list.append(client_temp)
        if item[21] == "yes":
            for j in range(0,5):
                list.append(client_temp)
        # client_temp.displayClient()
    csvFile.close()

#read a file;----------
def read_file_no_over(name,list):
    csvFile = open(name, "r")
    reader = csv.reader(csvFile)
    for item in reader:
        client_temp = Client(item[0],item[1],item[2],item[3],item[4],item[5],item[6],item[7],item[8],item[9],item[10],item[11],item[12],item[13],item[14],item[15],item[16],item[17],item[18],item[19],item[20],item[21])
        list.append(client_temp)
        # client_temp.displayClient()
    csvFile.close()



def write_file(name,list):
    # write a file;---------------
    file_object = open(name, 'w')
    file_object.write("id,"
                      "label,"
                      "age,"
                      "job,"
                      # "marital,"
                      "education,"
                      # "default,"
                      # "housing,"
                      # "loan,"
                      # "contact,"
                      # "month,"
                      "day_of_week,"
                      "duration,"
                      "campaign,"
                      "pdays,"
                      # "previous,"
                      # "poutcome,"
                      # "emp.var.rate,"
                      # "cons.price.idx,"
                      # "cons.conf.idx,"
                      "euribor3m,"
                      # "extra_feature,"
                      "nr.employed\n" )
    for item in range(len(list)):
        file_object.write("%s," % list[item].id)
        file_object.write("%s," % list[item].Output)

        file_object.write("%s," % list[item].age)
        file_object.write("%s," % list[item].job)
        # file_object.write("%s," % list[item].marital)
        file_object.write("%s," % list[item].education)
        # file_object.write("%s," % list[item].default )
        # file_object.write("%s," % list[item].housing)
        # file_object.write("%s," % list[item].loan)
        # file_object.write("%s," % list[item].contact)
        # file_object.write("%s," % list[item].month)
        file_object.write("%s," % list[item].day_of_week)
        file_object.write("%s," % list[item].duration)
        file_object.write("%s," % list[item].campaign)
        file_object.write("%s," % list[item].pdays)
        # file_object.write("%s," % list[item].previous)
        # file_object.write("%s," % list[item].poutcome )
        # file_object.write("%s," % list[item].emp_var_rate)
        # file_object.write("%s," % list[item].cons_price_idx)
        # file_object.write("%s," % list[item].cons_conf_idx)

        file_object.write("%s," % list[item].euribor3m)
        # file_object.write("%s," % list[item].extra_feature)

        file_object.write("%s\n" % list[item].nr_employed)
    file_object.close()
    # -----------------------------



read_file("originTrain_0.85.csv",train_list)
write_file("_Train_0.85.csv",train_list)

read_file_no_over("originTrain_0.15.csv",test_list)
write_file("_Train_0.15.csv",test_list)

read_file_no_over("originTest_truth.csv",test_true_list)
write_file("_Test_truth.csv",test_true_list)



