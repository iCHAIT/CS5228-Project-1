import csv
import string

def chackJOB(job):
    if job == "admin.": return  0
    if job == "self-employed": return  1
    if job == "technician": return  2
    if job == "management": return  3
    if job == "services": return  4
    if job == "student": return  5
    if job == "unemployed": return  6
    if job == "housemaid": return  7
    if job == "blue-collar": return  8
    if job == "entrepreneur": return  9
    if job == "retired": return  10
    if job == "unknown": return  11

def chackPRVE(num):
    if num == 0: return 0
    if num == 1: return 1
    if num == 2: return 2
    if num == 3: return 3
    if num == 4: return 4
    if num == 5: return 5
    if num == 6: return 6
    if num == 7: return 7

def chackEDUCATION(education):
    if education == "high.school": return  0
    if education == "university.degree": return  1
    if education == "basic.6y": return 2
    if education == "basic.4y": return  3
    if education == "basic.9y": return  4
    if education == "professional.course": return 5
    if education == "illiterate": return  6
    if education == "unknown": return 7

def chackDEFAULT(default):
    if default == "no": return 0
    if default == "yes": return 1
    if default == "unknown": return 2

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
    if day_of_week == "mon": return 0
    if day_of_week == "tue": return 1
    if day_of_week == "wed": return 2
    if day_of_week == "thu": return 3
    if day_of_week == "fri": return 4

def chackPdays(num):
    if num <= 8: return 0 # 5.88
    if num > 8 and num <= 16: return 1 # 5.08
    if num > 16 and num <= 30: return 2 # 4.74
    if num == 999: return 3 # 0.81

def chackAGE(num):
    if num <= 20: return 0
    if num > 20 and num <= 40: return 1
    if num > 40 and num <= 60: return 2
    if num > 60 and num <= 80: return 3
    if num > 80: return 4

def chackDURA(num):
    if num <= 60: return 0
    if num > 60 and num <= 100: return 1
    if num > 100 and num <= 150: return 2
    if num > 150 and num <= 200: return 3
    if num > 200 and num <= 250: return 4
    if num > 250 and num <= 300: return 5
    if num > 300 and num <= 400: return 6
    if num > 400 and num <= 450: return 7
    if num > 450 and num <= 500: return 8
    if num > 500 and num <= 550: return 9
    if num > 550 and num <= 600: return 10
    if num > 600 and num <= 800: return 11
    if num > 800: return 12


#######################################################################
def chackCONTACK(contact):
    if contact == "cellular": return 0 # 1.31;
    if contact == "telephone": return 1 # 0.46;

def chackHOUSE(housing):
    if housing == "no": return 0;
    if housing == "yes": return 1;
    if housing == "unknown": return 2;

def chackLOAN(loan):
    if loan == "no": return 0;
    if loan == "yes": return 1;
    if loan == "unknown": return 2;

def chackPOUTCOME(poutcome):
    if poutcome == "failure": return 0 # 1.26;
    if poutcome == "nonexistent": return 1 # 0.77;
    if poutcome == "success": return  2 #5.84;

def chackCAMP(num):
    if num <= 10: return 0 # 1.01
    if num > 10 and num <= 20: return 1 # 0.37
    if num > 20 and num <= 30: return 2 # 0.09
    if num > 30 and num <= 40: return 3 # 0
    if num > 40: return 1 # 0

def chackMARITAL(marital):
    if marital == "single": return 0;
    if marital == "divorced": return 1;
    if marital == "married": return 2;
    if marital == "unknown": return 3;


def chackOUTPUT(Output):
    if Output == "no": return 0
    if Output == "yes": return 1



train_client_list = [];  # a list, type of this list is Class Client, used to save all the information of all train data clients
test_client_list = [];  # a list, type of this list is Class Client, used to save all the information of all train data clients

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
        self.emp_var_rate = float(emp_var_rate)             # 16
        self.cons_price_idx = float(cons_price_idx)          # 17
        self.cons_conf_idx = float(cons_conf_idx)            # 18
        self.euribor3m = float(euribor3m)                    # 19
        self.nr_employed = float(nr_employed)              # 20
        self.Output = chackOUTPUT(Output)                    # 21
        self.grade = 0


    def displayClient(self):
        print (self.id, self.age, self.job, self.marital, self.education, self.default, self.housing, self.loan, self.contact, self.month, self.day_of_week, self.duration, self.campaign, self.pdays, self.previous, self.poutcome, self.emp_var_rate, self.cons_price_idx, self.cons_conf_idx, self.euribor3m, self.nr_employed, self.Output)



# read a train file;----------
csvFile = open("train.csv", "r")
reader = csv.reader(csvFile)
for item in reader:
    client_temp = Client(item[0],item[1],item[2],item[3],item[4],item[5],item[6],item[7],item[8],item[9],item[10],item[11],item[12],item[13],item[14],item[15],item[16],item[17],item[18],item[19],item[20],item[21])
    train_client_list.append(client_temp)
    # client_temp.displayClient()
csvFile.close()

# read a test file;----------
csvFile = open("test.csv", "r")
reader = csv.reader(csvFile)
for item in reader:
    client_temp = Client(item[0],item[1],item[2],item[3],item[4],item[5],item[6],item[7],item[8],item[9],item[10],item[11],item[12],item[13],item[14],item[15],item[16],item[17],item[18],item[19],item[20],"no")
    test_client_list.append(client_temp)
    client_temp.displayClient()
csvFile.close()


# write a file;---------------
file_object = open('train_song.txt', 'w')
# file_object.write("id,prediction\n" )
for item in range(len(train_client_list)):
    # test_client_list[item].displayClient()
    file_object.write("%s " % train_client_list[item].Output )
    file_object.write("1:%s " % train_client_list[item].age )
    file_object.write("2:%s " % train_client_list[item].job)
    file_object.write("3:%s " % train_client_list[item].marital)
    file_object.write("4:%s " % train_client_list[item].education)
    file_object.write("5:%s " % train_client_list[item].default)
    file_object.write("6:%s " % train_client_list[item].housing)
    file_object.write("7:%s " % train_client_list[item].loan)
    file_object.write("8:%s " % train_client_list[item].contact)
    file_object.write("9:%s " % train_client_list[item].month)
    file_object.write("10:%s " % train_client_list[item].day_of_week)
    file_object.write("11:%s " % train_client_list[item].duration)
    file_object.write("12:%s " % train_client_list[item].campaign)
    file_object.write("13:%s " % train_client_list[item].pdays)
    file_object.write("14:%s " % train_client_list[item].previous)
    file_object.write("15:%s " % train_client_list[item].poutcome)
    file_object.write("16:%s " % train_client_list[item].emp_var_rate)
    file_object.write("17:%s " % train_client_list[item].cons_price_idx)
    file_object.write("18:%s " % train_client_list[item].cons_conf_idx)
    file_object.write("19:%s " % train_client_list[item].euribor3m)
    file_object.write("20:%s \n" % train_client_list[item].nr_employed)
file_object.close()
#-----------------------------


# write a file;---------------
file_object = open('test_song.txt', 'w')
# file_object.write("id,prediction\n" )
for item in range(len(test_client_list)):
    # test_client_list[item].displayClient()
    file_object.write("%s " % test_client_list[item].Output )
    file_object.write("1:%s " % test_client_list[item].age )
    file_object.write("2:%s " % test_client_list[item].job)
    file_object.write("3:%s " % test_client_list[item].marital)
    file_object.write("4:%s " % test_client_list[item].education)
    file_object.write("5:%s " % test_client_list[item].default)
    file_object.write("6:%s " % test_client_list[item].housing)
    file_object.write("7:%s " % test_client_list[item].loan)
    file_object.write("8:%s " % test_client_list[item].contact)
    file_object.write("9:%s " % test_client_list[item].month)
    file_object.write("10:%s " % test_client_list[item].day_of_week)
    file_object.write("11:%s " % test_client_list[item].duration)
    file_object.write("12:%s " % test_client_list[item].campaign)
    file_object.write("13:%s " % test_client_list[item].pdays)
    file_object.write("14:%s " % test_client_list[item].previous)
    file_object.write("15:%s " % test_client_list[item].poutcome)
    file_object.write("16:%s " % test_client_list[item].emp_var_rate)
    file_object.write("17:%s " % test_client_list[item].cons_price_idx)
    file_object.write("18:%s " % test_client_list[item].cons_conf_idx)
    file_object.write("19:%s " % test_client_list[item].euribor3m)
    file_object.write("20:%s \n" % test_client_list[item].nr_employed)
file_object.close()
#-----------------------------

