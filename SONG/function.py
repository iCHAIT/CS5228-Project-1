import csv
import string

def chackJOB(job):
    if job == "admin.": return  1.18
    if job == "self-employed": return  0.95
    if job == "technician": return  0.94
    if job == "management": return  1.01
    if job == "services": return  0.75
    if job == "student": return  2.79
    if job == "unemployed": return  1.18
    if job == "housemaid": return  0.84
    if job == "blue-collar": return  0.58
    if job == "entrepreneur": return  0.77
    if job == "retired": return  2.26
    if job == "unknown": return  1.13

def chackPRVE(num):
    if num == 0: return 0.77
    if num == 1: return 1.92
    if num == 2: return 4.02
    if num == 3: return 5.26
    if num == 4: return 5.4
    if num == 5: return 6.6
    if num == 6: return 6
    if num == 7: return 0

def chackEDUCATION(education):
    if education == "high.school": return  0.96;
    if education == "university.degree": return  1.24;
    if education == "basic.6y": return 0.75;
    if education == "basic.4y": return  0.88;
    if education == "basic.9y": return  0.66;
    if education == "professional.course": return 1;
    if education == "illiterate": return  2.77;
    if education == "unknown": return 1.25;

def chackDEFAULT(default):
    if default == "no": return 0.7;
    if default == "yes": return -1;
    if default == "unknown": return 0.45;

def chackMOUTH(month):
    if month == "may": return 0.56
    if month == "nov": return 0.91
    if month == "apr": return 1.81
    if month == "aug": return 0.96
    if month == "sep": return 3.97
    if month == "jun": return 0.92
    if month == "oct": return 4
    if month == "jul": return 0.82
    if month == "dec": return 4.65
    if month == "mar": return 4.51

def chackWEEK(day_of_week):
    if day_of_week == "mon": return 0.88
    if day_of_week == "tue": return 1.04
    if day_of_week == "wed": return 1.02
    if day_of_week == "thu": return 1.10
    if day_of_week == "fri": return 0.95

def chackPdays(num):
    if num <= 8: return 1 # 5.88
    if num > 8 and num <= 16: return 1 # 5.08
    if num > 16 and num <= 30: return 1 # 4.74
    if num == 999: return 0.81 # 0.81

def chackAGE(num):
    if num <= 20: return 3.5
    if num > 20 and num <= 40: return 1
    if num > 40 and num <= 60: return 0.81
    if num > 60 and num <= 80: return 4.06
    if num > 80: return 4.35

def chackDURA(num):
    if num <= 60: return 0
    if num > 60 and num <= 100: return 1.12
    if num > 100 and num <= 150: return 0.34
    if num > 150 and num <= 200: return 0.57
    if num > 200 and num <= 250: return 0.84
    if num > 250 and num <= 300: return 1.15
    if num > 300 and num <= 400: return 1.28
    if num > 400 and num <= 450: return 1.64
    if num > 450 and num <= 500: return 2
    if num > 500 and num <= 550: return 2.17
    if num > 550 and num <= 600: return 2.8
    if num > 600 and num <= 800: return 3.5
    if num > 800: return 5.13


#######################################################################
def chackCONTACK(contact):
    if contact == "cellular": return 1 # 1.31;
    if contact == "telephone": return 1 # 0.46;

def chackHOUSE(housing):
    if housing == "no": return 1;
    if housing == "yes": return 1;
    if housing == "unknown": return 1;

def chackLOAN(loan):
    if loan == "no": return 1;
    if loan == "yes": return 1;
    if loan == "unknown": return 1;

def chackPOUTCOME(poutcome):
    if poutcome == "failure": return 1 # 1.26;
    if poutcome == "nonexistent": return 1 # 0.77;
    if poutcome == "success": return  1 #5.84;

def chackCAMP(num):
    if num <= 10: return 1 # 1.01
    if num > 10 and num <= 20: return 1 # 0.37
    if num > 20 and num <= 30: return 1 # 0.09
    if num > 30 and num <= 40: return 1 # 0
    if num > 40: return 1 # 0

def chackMARITAL(marital):
    if marital == "single": return 1;
    if marital == "divorced": return 1;
    if marital == "married": return 1;
    if marital == "unknown": return 1;

def chackEMP(num):
    if num <= -2: return 1 # 0.56
    if num > -2 and num <= 0: return 1 # 1.48
    if num > 0 and num <= 2: return 1 # 0.41
def chackPRICE(num):
    if num < 93: return 1 # 1.78
    if num >= 93  and num < 94: return 1 # 0.65
    if num >= 94: return 1 # 1.37
def chackIDX(num):
    if num < -43: return 1 # 1.22
    if num >= -43  and num < -34: return 1 # 0.73
    if num >= -34: return 1 # 3.64
def chackEURIBOR(num):
    if num < 1: return 1 # 4.13
    if num >= 1  and num < 2: return 1 # 1.39
    if num >= 2 and num < 4: return 1 # 0
    if num >= 4: return 1 # 0.43
def chackNR_EM(num):
    if num < 5052: return 1 # 4.39
    if num >= 5052  and num < 5141: return 1 # 1.46
    if num >= 5141: return 1 # 0.42

def chackOUTPUT(Output):
    if Output == "no": return 0
    if Output == "yes": return 1



train_client_list_1 = [];  # a list, type of this list is Class Client, used to save all the information of all train data clients
train_client_list_0 = [];  # a list, type of this list is Class Client, used to save all the information of all train data clients

test_client_list = [];
w = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1];
class Client: # a class to of client, used to save all the data.

    def __init__(self, id, age, job, marital, education, default, housing, loan, contact, month, day_of_week, duration, campaign, pdays, previous, poutcome, emp_var_rate, cons_price_idx, cons_conf_idx, euribor3m, nr_employed, Output):
        self.id = int(id)                                         # 0
        self.age = chackAGE(int(age))                                  # 1
        self.job = chackJOB(job)                             # 2
        self.marital = chackMARITAL(marital)                 # 3
        self.education = chackEDUCATION(education)           # 4
        self.default = chackDEFAULT(default)                 # 5
        self.housing = chackHOUSE(housing)                   # 6
        self.loan = chackLOAN(loan)                          # 7
        self.contact = chackCONTACK(contact)                 # 8
        self.month = chackMOUTH(month)                       # 9
        self.day_of_week = chackWEEK(day_of_week)            # 10
        self.duration = chackDURA(int(duration))                        # 11
        self.campaign = chackCAMP(int(campaign))                        # 12
        self.pdays = chackPdays(int(pdays))                              # 13
        self.previous = chackPRVE(int(previous))                        # 14
        self.poutcome = chackPOUTCOME(poutcome)              # 15
        self.emp_var_rate = chackEMP(float(emp_var_rate))              # 16
        self.cons_price_idx = chackPRICE(float(cons_price_idx))          # 17
        self.cons_conf_idx = chackIDX(float(cons_conf_idx))            # 18
        self.euribor3m = chackEURIBOR(float(euribor3m))                    # 19
        self.nr_employed = chackNR_EM(float(nr_employed))                # 20
        self.Output = chackOUTPUT(Output)                    # 21
        self.grade = 0


    def displayClient(self):
        print (self.id, self.age, self.job, self.marital, self.education, self.default, self.housing, self.loan, self.contact, self.month, self.day_of_week, self.duration, self.campaign, self.pdays, self.previous, self.poutcome, self.emp_var_rate, self.cons_price_idx, self.cons_conf_idx, self.euribor3m, self.nr_employed, self.Output)



# read a train file;----------
csvFile = open("train.csv", "r")
reader = csv.reader(csvFile)
for item in reader:
    client_temp = Client(item[0],item[1],item[2],item[3],item[4],item[5],item[6],item[7],item[8],item[9],item[10],item[11],item[12],item[13],item[14],item[15],item[16],item[17],item[18],item[19],item[20],item[21])
    if item[21] == "yes":
        train_client_list_1.append(client_temp)
        # client_temp.displayClient()
    if item[21] == "no":
        train_client_list_0.append(client_temp)
csvFile.close()
# ----------------------------

# ----------------------------


# read a test file;-----------
csvFile = open("train.csv", "r")
# csvFile = open("test.csv", "r")

reader = csv.reader(csvFile)
for item in reader:
    client_temp = Client(item[0],item[1],item[2],item[3],item[4],item[5],item[6],item[7],item[8],item[9],item[10],item[11],item[12],item[13],item[14],item[15],item[16],item[17],item[18],item[19],item[20],0)
    # client_temp.displayClient()  #  for test
    test_client_list.append(client_temp)
csvFile.close()
# ----------------------------

# ----------------------------
# ----------------------------
sign = 20;
def compute(item):
    temp = w[0] * item.age + w[1] * item.job + w[2] * item.marital + w[3] * item.education + w[4] * item.default + w[5] * item.housing + w[6] * item.loan + w[7] * item.contact + w[8] * item.month + w[9] * item.day_of_week + w[10] * item.duration + w[11] * item.campaign + w[12] * item.pdays + w[13] * item.previous + w[14] * item.poutcome + w[15] * item.emp_var_rate + w[16] * item.cons_price_idx + w[17] * item.cons_conf_idx + w[18] * item.euribor3m + w[19] * item.nr_employed
    # ----------------------------
    if item.job == "retired" and item.poutcome == "success":
        temp = temp + 2

    # ----------------------------
    item.grade = temp
    if temp > sign:
        item.Output = 1;
    if temp <= sign:
        item.Output = 0;


# write a file;---------------
file_object = open('sampleSubmission1.csv', 'w')
# file_object.write("id,prediction\n" )
for item in range(len(test_client_list)):
    # test_client_list[item].displayClient()
    compute(test_client_list[item])
    file_object.write("%s," % test_client_list[item].id )
    file_object.write("%s," % test_client_list[item].Output )
    file_object.write("%s\n" % test_client_list[item].grade)
file_object.close()
#-----------------------------

