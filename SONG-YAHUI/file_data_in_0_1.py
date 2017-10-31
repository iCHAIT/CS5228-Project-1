import csv
import string

########################Categorical################################

def chackJOB(Job,job):
    if job == "admin.": Job[0] = 1
    if job == "self-employed": Job[1] = 1
    if job == "technician": Job[2] = 1
    if job == "management": Job[3] = 1
    if job == "services": Job[4]= 1
    if job == "student": Job[5] = 1
    if job == "unemployed": Job[6] = 1
    if job == "housemaid": Job[7] = 1
    if job == "blue-collar": Job[8] = 1
    if job == "entrepreneur": Job[9] = 1
    if job == "retired": Job[10] = 1
    if job == "unknown": Job[11] = 1

def chackEDUCATION(Education,education):
    if education == "high.school": Education[0] = 1
    if education == "university.degree": Education[1] = 1
    if education == "basic.6y": Education[2] = 1
    if education == "basic.4y": Education[3] = 1
    if education == "basic.9y": Education[4] = 1
    if education == "professional.course": Education[5] = 1
    if education == "illiterate": Education[6] = 1
    if education == "unknown": Education[7] = 1

def chackDEFAULT(Default,default):
    if default == "no": Default[0] = 1
    if default == "yes": Default[1] = 1
    if default == "unknown": Default[2] = 1

def chackMOUTH(Month,month):
    if month == "may": Month[0] = 1
    if month == "nov": Month[1] = 1
    if month == "apr": Month[ 2] = 1
    if month == "aug": Month[ 3] = 1
    if month == "sep": Month[ 4] = 1
    if month == "jun": Month[ 5] = 1
    if month == "oct": Month[ 6] = 1
    if month == "jul": Month[ 7] = 1
    if month == "dec": Month[ 8] = 1
    if month == "mar": Month[ 9] = 1

def chackWEEK(Day_of_week,day_of_week):
    if day_of_week == "mon": Day_of_week[0] = 1
    if day_of_week == "tue": Day_of_week[ 1] = 1
    if day_of_week == "wed": Day_of_week[ 2] = 1
    if day_of_week == "thu": Day_of_week[ 3] = 1
    if day_of_week == "fri": Day_of_week[ 4] = 1

def chackCONTACK(contact):
    if contact == "cellular": return 0 # 1.31;
    if contact == "telephone": return 1 # 0.46;

def chackHOUSE(Housing,housing):
    if housing == "no": Housing[0]=1
    if housing == "yes": Housing[1]=1
    if housing == "unknown": Housing[2]=1

def chackLOAN(Loan,loan):
    if loan == "no": Loan[0]=1
    if loan == "yes": Loan[1]=1
    if loan == "unknown": Loan[2]=1

def chackPOUTCOME(Poutcome,poutcome):
    if poutcome == "failure": Poutcome[0]=1 # 1.26;
    if poutcome == "nonexistent": Poutcome[1]=1 # 0.77;
    if poutcome == "success": Poutcome[2] = 1 #5.84;

def chackMARITAL(Marital,marital):
    if marital == "single": Marital[0] =1
    if marital == "divorced": Marital[1]=1
    if marital == "married": Marital[2]=1
    if marital == "unknown": Marital[3] = 1


################Numeric#####################################


def chackPdays(num):
    # return num
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

############################################################################################

def chackOUTPUT(Output):
    if Output == "no": return 0
    if Output == "yes": return 1



train_client_list = [];  # a list, type of this list is Class Client, used to save all the information of all train data clients
test_client_list = [];  # a list, type of this list is Class Client, used to save all the information of all train data clients

class Client: # a class to of client, used to save all the data.

    def __init__(self, id, age, job, marital, education, default, housing, loan, contact, month, day_of_week, duration, campaign, pdays, previous, poutcome, emp_var_rate, cons_price_idx, cons_conf_idx, euribor3m, nr_employed, Output):
        self.id = id                                         # 0
        ########################Categorical################################
        self.Job = [0,0,0,0,0,0,0,0,0,0,0,0]; chackJOB(self.Job,job)                            # 2
        self.Marital = [0,0,0,0]; chackMARITAL(self.Marital,marital)                 # 3
        self.Education = [0,0,0,0,0,0,0,0]; chackEDUCATION(self.Education,education)           # 4
        self.Default =[0,0,0]; chackDEFAULT(self.Default,default)                 # 5
        self.Housing = [0,0,0]; chackHOUSE(self.Housing,housing)                   # 6
        self.Loan = [0,0,0]; chackLOAN(self.Loan, loan)                          # 7
        self.contact = chackCONTACK(contact)  # 8
        self.Month = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];chackMOUTH(self.Month, month)  # 9
        self.Day_of_week = [0, 0, 0, 0, 0];chackWEEK(self.Day_of_week, day_of_week)  # 10
        self.Poutcome = [0, 0, 0];chackPOUTCOME(self.Poutcome, poutcome)  # 15
        ################Numeric#####################################
        self.age = (int(age)-17)/81  # 1
        self.duration = chackDURA(int(duration))/4199                       # 11
        self.campaign = chackCAMP(int(campaign)) /56                       # 12
        self.pdays = chackPdays(int(pdays))                              # 13
        self.previous = int(previous)/7                     # 14
        self.emp_var_rate = (float(emp_var_rate) + 3.4) /4.8           # 16
        self.cons_price_idx = (float(cons_price_idx) - 92.201)/2.6         # 17
        self.cons_conf_idx = (float(cons_conf_idx) + 50.8)/24          # 18
        self.euribor3m = (float(euribor3m) -0.634)/ 4.411           # 19
        self.nr_employed = (float(nr_employed)-4963) /265           # 20

        self.Output = chackOUTPUT(Output)                    # 21


    def displayClient(self):
        print (\
            self.id, self.age, self.Job, self.Marital\
            , self.Education, self.Default, self.Housing\
            , self.Loan, self.contact, self.Month, self.Day_of_week\
            , self.duration, self.campaign, self.pdays, self.previous\
            , self.Poutcome, self.emp_var_rate, self.cons_price_idx\
            , self.cons_conf_idx, self.euribor3m, self.nr_employed)



# read a train file;----------
csvFile = open("train.csv", "r")
reader = csv.reader(csvFile)
for item in reader:
    client_temp = Client(item[0],item[1],item[2],item[3],item[4],item[5],item[6],item[7],item[8],item[9],item[10],item[11],item[12],item[13],item[14],item[15],item[16],item[17],item[18],item[19],item[20],item[21])
    train_client_list.append(client_temp)
    # client_temp.displayClient()
csvFile.close()


# write a file;---------------
file_object = open('train_song.csv', 'w')
file_object.write("label,age,\
job0,job1,job2,job3,job4,job5,job6,job7,job8,job9,job10,job11,\
marital0,marital1,marital2,marital3,\
education0,education1,education2,education3,education4,education5,education6,education7,\
default0,default1,default2,\
housing0,housing1,housing2,\
loan0,loan1,loan2,\
contact,\
month0,month1,month2,month3,month4,month5,month6,month7,month8,month9,\
day_of_week0,day_of_week1,day_of_week2,day_of_week3,day_of_week4,\
duration,campaign,pdays,previous,\
poutcome0,poutcome1,poutcome2,\
emp.var.rate,cons.price.idx,cons.conf.idx,euribor3m,nr.employed\n" )
for item in range(len(train_client_list)):
    # test_client_list[item].displayClient()
    file_object.write("%s," % train_client_list[item].Output )
    file_object.write("%s," % train_client_list[item].age )

    for j in range(0,len(train_client_list[item].Job)):file_object.write("%s," % train_client_list[item].Job[j])

    for j in range(0, len(train_client_list[item].Marital)):file_object.write("%s," % train_client_list[item].Marital[j])

    for j in range(0, len(train_client_list[item].Education)):file_object.write("%s," % train_client_list[item].Education[j])

    for j in range(0, len(train_client_list[item].Default)):file_object.write("%s," % train_client_list[item].Default[j])

    for j in range(0, len(train_client_list[item].Housing)):file_object.write("%s," % train_client_list[item].Housing[j])

    for j in range(0, len(train_client_list[item].Loan)):file_object.write("%s," % train_client_list[item].Loan[j])

    file_object.write("%s," % train_client_list[item].contact)

    for j in range(0, len(train_client_list[item].Month)):file_object.write("%s," % train_client_list[item].Month[j])

    for j in range(0, len(train_client_list[item].Day_of_week)):file_object.write("%s," % train_client_list[item].Day_of_week[j])


    file_object.write("%s," % train_client_list[item].duration)
    file_object.write("%s," % train_client_list[item].campaign)
    file_object.write("%s," % train_client_list[item].pdays)
    file_object.write("%s," % train_client_list[item].previous)

    for j in range(0, len(train_client_list[item].Poutcome)):file_object.write("%s," % train_client_list[item].Poutcome[j])

    file_object.write("%s," % train_client_list[item].emp_var_rate)
    file_object.write("%s," % train_client_list[item].cons_price_idx)
    file_object.write("%s," % train_client_list[item].cons_conf_idx)
    file_object.write("%s," % train_client_list[item].euribor3m)
    file_object.write("%s\n" % train_client_list[item].nr_employed)
file_object.close()
#-----------------------------




###########test

# read a train file;----------
csvFile = open("test.csv", "r")
reader = csv.reader(csvFile)
for item in reader:
    client_temp = Client(item[0],item[1],item[2],item[3],item[4],item[5],item[6],item[7],item[8],item[9],item[10],item[11],item[12],item[13],item[14],item[15],item[16],item[17],item[18],item[19],item[20],"yes")
    test_client_list.append(client_temp)
    client_temp.displayClient()
csvFile.close()


# write a file;---------------
file_object = open('test_song.csv', 'w')
file_object.write("label,age,\
job0,job1,job2,job3,job4,job5,job6,job7,job8,job9,job10,job11,\
marital0,marital1,marital2,marital3,\
education0,education1,education2,education3,education4,education5,education6,education7,\
default0,default1,default2,\
housing0,housing1,housing2,\
loan0,loan1,loan2,\
contact,\
month0,month1,month2,month3,month4,month5,month6,month7,month8,month9,\
day_of_week0,day_of_week1,day_of_week2,day_of_week3,day_of_week4,\
duration,campaign,pdays,previous,\
poutcome0,poutcome1,poutcome2,\
emp.var.rate,cons.price.idx,cons.conf.idx,euribor3m,nr.employed\n" )
for item in range(len(test_client_list)):
    # test_client_list[item].displayClient()
    file_object.write("%s," % test_client_list[item].Output )
    file_object.write("%s," % test_client_list[item].age )

    for j in range(0, len(test_client_list[item].Job)): file_object.write("%s," % test_client_list[item].Job[j])

    for j in range(0, len(test_client_list[item].Marital)): file_object.write(
        "%s," % test_client_list[item].Marital[j])

    for j in range(0, len(test_client_list[item].Education)): file_object.write(
        "%s," % test_client_list[item].Education[j])

    for j in range(0, len(test_client_list[item].Default)): file_object.write(
        "%s," % test_client_list[item].Default[j])

    for j in range(0, len(test_client_list[item].Housing)): file_object.write(
        "%s," % test_client_list[item].Housing[j])

    for j in range(0, len(test_client_list[item].Loan)): file_object.write("%s," % test_client_list[item].Loan[j])

    file_object.write("%s," % test_client_list[item].contact)

    for j in range(0, len(test_client_list[item].Month)): file_object.write("%s," % test_client_list[item].Month[j])

    for j in range(0, len(test_client_list[item].Day_of_week)): file_object.write(
        "%s," % test_client_list[item].Day_of_week[j])

    file_object.write("%s," % test_client_list[item].duration)
    file_object.write("%s," % test_client_list[item].campaign)
    file_object.write("%s," % test_client_list[item].pdays)
    file_object.write("%s," % test_client_list[item].previous)

    for j in range(0, len(test_client_list[item].Poutcome)): file_object.write(
        "%s," % test_client_list[item].Poutcome[j])

    file_object.write("%s," % test_client_list[item].emp_var_rate)
    file_object.write("%s," % test_client_list[item].cons_price_idx)
    file_object.write("%s," % test_client_list[item].cons_conf_idx)
    file_object.write("%s," % test_client_list[item].euribor3m)
    file_object.write("%s\n" % test_client_list[item].nr_employed)
file_object.close()
#-----------------------------