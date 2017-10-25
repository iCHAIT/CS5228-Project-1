import csv
import string

def chackOUTPUT(Output):
    if Output == "no": return 0
    if Output == "yes": return 1
    return Output



test_client_list = [];

w = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1];
class Client: # a class to of client, used to save all the data.

    def __init__(self, id, age, job, marital, education, default, housing, loan, contact, month, day_of_week, duration, campaign, pdays, previous, poutcome, emp_var_rate, cons_price_idx, cons_conf_idx, euribor3m, nr_employed, Output):
        self.id = int(id)                                    # 0
        self.age = int(age)                                  # 1
        self.job = job                                       # 2
        self.marital = marital                               # 3
        self.education = education                           # 4
        self.default = default                               # 5
        self.housing = housing                               # 6
        self.loan = loan                                     # 7
        self.contact = contact                               # 8
        self.month = month                                   # 9
        self.day_of_week = day_of_week                       # 10
        self.duration = int(duration)                        # 11
        self.campaign = int(campaign)                        # 12
        self.pdays = int(pdays)                              # 13
        self.previous = int(previous)                        # 14
        self.poutcome = poutcome              # 15
        self.emp_var_rate = float(emp_var_rate)              # 16
        self.cons_price_idx = float(cons_price_idx)          # 17
        self.cons_conf_idx = float(cons_conf_idx)            # 18
        self.euribor3m = float(euribor3m)                    # 19
        self.nr_employed = float(nr_employed)                # 20
        self.Output = chackOUTPUT(Output)                    # 21


    def displayClient(self):
        print (self.id, self.age, self.job, self.marital, self.education, self.default, self.housing, self.loan, self.contact, self.month, self.day_of_week, self.duration, self.campaign, self.pdays, self.previous, self.poutcome, self.emp_var_rate, self.cons_price_idx, self.cons_conf_idx, self.euribor3m, self.nr_employed, self.Output)



# read a test file;-----------
csvFile = open("train.csv", "r")
# csvFile = open("test.csv", "r")

reader = csv.reader(csvFile)
for item in reader:
    client_temp = Client(item[0],item[1],item[2],item[3],item[4],item[5],item[6],item[7],item[8],item[9],item[10],item[11],item[12],item[13],item[14],item[15],item[16],item[17],item[18],item[19],item[20],item[21])
    # client_temp.displayClient()  #  for test
    test_client_list.append(client_temp)
csvFile.close()


count_yes = 0;
for i in range(0,len(test_client_list)):
    if test_client_list[i].Output == 1:
        count_yes  = count_yes + 1;

print("COUNT_YES_IN_TOTAL:",count_yes,"ratio:", count_yes/len(test_client_list))
ration = count_yes/len(test_client_list);



print("AGE:")
count0_20 = 0;count0_20_total = 0;
count20_40 = 0;count20_40_total = 0;
count40_60 = 0;count40_60_total = 0;
count60_80 = 0;count60_80_total = 0;
count80=0;count80_total =0;
for i in range(0,len(test_client_list)):
    if test_client_list[i].age <= 20: count0_20_total = count0_20_total + 1;
    if test_client_list[i].age > 20 and test_client_list[i].age <= 40: count20_40_total = count20_40_total + 1;
    if test_client_list[i].age > 40 and test_client_list[i].age <= 60: count40_60_total = count40_60_total + 1;
    if test_client_list[i].age > 60 and test_client_list[i].age <= 80: count60_80_total = count60_80_total + 1;
    if test_client_list[i].age > 80: count80_total = count80_total + 1;

    if test_client_list[i].Output == 1:
        if test_client_list[i].age <= 20: count0_20 = count0_20 +1;
        if test_client_list[i].age > 20 and test_client_list[i].age <= 40: count20_40 = count20_40 +1;
        if test_client_list[i].age > 40 and test_client_list[i].age <= 60: count40_60 = count40_60 +1;
        if test_client_list[i].age > 60 and test_client_list[i].age <= 80: count60_80 = count60_80 +1;
        if test_client_list[i].age > 80: count80 = count80 +1;
# print(count0_20,count20_40,count40_60,count60_80,count80)
# print(count0_20/count0_20_total,count20_40/count20_40_total,count40_60/count40_60_total,count60_80/count60_80_total,count80/count80_total)
print(count0_20/count0_20_total/ration,count20_40/count20_40_total/ration,count40_60/count40_60_total/ration,count60_80/count60_80_total/ration,count80/count80_total/ration)

'''
print("JOB:")
list_job_total =[0,0,0,0,0,0,0,0,0,0,0,0]
list_job =[0,0,0,0,0,0,0,0,0,0,0,0]
for i in range(0,len(test_client_list)):
    if test_client_list[i].job == "admin.": list_job_total[0] = list_job_total[0] +1
    if test_client_list[i].job == "self-employed": list_job_total[1] = list_job_total[1] +1
    if test_client_list[i].job == "technician": list_job_total[2] = list_job_total[2] +1
    if test_client_list[i].job == "management": list_job_total[3] = list_job_total[3] +1
    if test_client_list[i].job == "services": list_job_total[4] = list_job_total[4] +1
    if test_client_list[i].job == "student": list_job_total[5] = list_job_total[5] +1
    if test_client_list[i].job == "unemployed": list_job_total[6] = list_job_total[6] +1
    if test_client_list[i].job == "housemaid": list_job_total[7] = list_job_total[7] +1
    if test_client_list[i].job == "blue-collar": list_job_total[8] = list_job_total[8] +1
    if test_client_list[i].job == "entrepreneur": list_job_total[9] = list_job_total[9] +1
    if test_client_list[i].job == "retired": list_job_total[10] = list_job_total[10] +1
    if test_client_list[i].job == "unknown": list_job_total[11] = list_job_total[11] +1
    if test_client_list[i].Output == 1:
        if test_client_list[i].job == "admin.": list_job[0] = list_job[0] + 1
        if test_client_list[i].job == "self-employed": list_job[1] = list_job[1] + 1
        if test_client_list[i].job == "technician": list_job[2] = list_job[2] + 1
        if test_client_list[i].job == "management": list_job[3] = list_job[3] + 1
        if test_client_list[i].job == "services": list_job[4] = list_job[4] + 1
        if test_client_list[i].job == "student": list_job[5] = list_job[5] + 1
        if test_client_list[i].job == "unemployed": list_job[6] = list_job[6] + 1
        if test_client_list[i].job == "housemaid": list_job[7] = list_job[7] + 1
        if test_client_list[i].job == "blue-collar": list_job[8] = list_job[8] + 1
        if test_client_list[i].job == "entrepreneur": list_job[9] = list_job[9] + 1
        if test_client_list[i].job == "retired": list_job[10] = list_job[10] + 1
        if test_client_list[i].job == "unknown": list_job[11] = list_job[11] + 1
for j in range(0,len(list_job_total)):
    # print(list_job[j])
    # print(list_job[j]/list_job_total[j])
    print(list_job[j]/list_job_total[j]/ration)
'''
'''
print("marital:")
list_marital_total =[0,0,0,0]
list_marital =[0,0,0,0]
for i in range(0,len(test_client_list)):
    if test_client_list[i].marital == "single": list_marital_total[0] = list_marital_total[0] +1
    if test_client_list[i].marital == "divorced": list_marital_total[1] = list_marital_total[1] +1
    if test_client_list[i].marital == "married": list_marital_total[2] = list_marital_total[2] +1
    if test_client_list[i].marital == "unknown": list_marital_total[3] = list_marital_total[3] +1
    if test_client_list[i].Output == 1:
        if test_client_list[i].marital == "single": list_marital[0] = list_marital[0] + 1
        if test_client_list[i].marital == "divorced": list_marital[1] = list_marital[1] + 1
        if test_client_list[i].marital == "married": list_marital[2] = list_marital[2] + 1
        if test_client_list[i].marital == "unknown": list_marital[3] = list_marital[3] + 1
for j in range(0,len(list_marital)):
    # print(list_marital[j])
    # print(list_marital[j]/list_marital_total[j])
    print(list_marital[j]/list_marital_total[j]/ration)
'''
'''
print("education:")
list_education_total =[0,0,0,0,0,0,0,0]
list_education =[0,0,0,0,0,0,0,0]
for i in range(0,len(test_client_list)):
    if test_client_list[i].education == "high.school": list_education_total[0] = list_education_total[0] +1
    if test_client_list[i].education == "university.degree": list_education_total[1] = list_education_total[1] +1
    if test_client_list[i].education == "basic.6y": list_education_total[2] = list_education_total[2] +1
    if test_client_list[i].education == "basic.4y": list_education_total[3] = list_education_total[3] +1
    if test_client_list[i].education == "basic.9y": list_education_total[4] = list_education_total[4] +1
    if test_client_list[i].education == "professional.course": list_education_total[5] = list_education_total[5] +1
    if test_client_list[i].education == "illiterate": list_education_total[6] = list_education_total[6] +1
    if test_client_list[i].education == "unknown": list_education_total[7] = list_education_total[7] +1
    if test_client_list[i].Output == 1:
        if test_client_list[i].education == "high.school": list_education[0] = list_education[0] + 1
        if test_client_list[i].education == "university.degree": list_education[1] = list_education[1] + 1
        if test_client_list[i].education == "basic.6y": list_education[2] = list_education[2] + 1
        if test_client_list[i].education == "basic.4y": list_education[3] = list_education[3] + 1
        if test_client_list[i].education == "basic.9y": list_education[4] = list_education[4] + 1
        if test_client_list[i].education == "professional.course": list_education[5] = list_education[5] + 1
        if test_client_list[i].education == "illiterate": list_education[6] = list_education[6] + 1
        if test_client_list[i].education == "unknown": list_education[7] = list_education[7] + 1
for j in range(0,len(list_education_total)):
    # print(list_education[j])
    # print(list_education[j]/list_education_total[j])
    print(list_education[j]/list_education_total[j]/ration)
'''

'''
print("default:")
list_default_total =[0,0,0]
list_default =[0,0,0]
for i in range(0,len(test_client_list)):
    if test_client_list[i].default == "no": list_default_total[0] = list_default_total[0] + 1
    if test_client_list[i].default == "yes": list_default_total[1] = list_default_total[1] + 1;
    if test_client_list[i].default == "unknown": list_default_total[2] = list_default_total[2] + 1
    if test_client_list[i].Output == 1:
        if test_client_list[i].default == "no": list_default[0] = list_default[0] + 1
        if test_client_list[i].default == "yes": list_default[1] = list_default[1] + 1
        if test_client_list[i].default == "unknown": list_default[2] = list_default[2] + 1
for j in range(0,len(list_default_total)):
    # print(list_default[j])
    # sprint(list_default[j]/list_default_total[j])
    print(list_default[j]/list_default_total[j]/ration)
'''

'''
print("housing:")
list_housing_total =[0,0,0]
list_housing =[0,0,0]
for i in range(0,len(test_client_list)):
    if test_client_list[i].housing == "no": list_housing_total[0] = list_housing_total[0] + 1
    if test_client_list[i].housing == "yes": list_housing_total[1] = list_housing_total[1] + 1
    if test_client_list[i].housing == "unknown": list_housing_total[2] = list_housing_total[2] + 1
    if test_client_list[i].Output == 1:
        if test_client_list[i].housing == "no": list_housing[0] = list_housing[0] + 1
        if test_client_list[i].housing == "yes": list_housing[1] = list_housing[1] + 1
        if test_client_list[i].housing == "unknown": list_housing[2] = list_housing[2] + 1
for j in range(0,len(list_housing_total)):
    # print(list_housing[j])
    # sprint(list_housing[j]/list_housing_total[j])
    print(list_housing[j]/list_housing_total[j]/ration)
'''

'''
print("loan:")
list_loan_total =[0,0,0]
list_loan =[0,0,0]
for i in range(0,len(test_client_list)):
    if test_client_list[i].loan == "no": list_loan_total[0] = list_loan_total[0] + 1
    if test_client_list[i].loan == "yes": list_loan_total[1] = list_loan_total[1] + 1
    if test_client_list[i].loan == "unknown": list_loan_total[2] = list_loan_total[2] + 1
    if test_client_list[i].Output == 1:
        if test_client_list[i].loan == "no": list_loan[0] = list_loan[0] + 1
        if test_client_list[i].loan == "yes": list_loan[1] = list_loan[1] + 1
        if test_client_list[i].loan == "unknown": list_loan[2] = list_loan[2] + 1
for j in range(0,len(list_loan_total)):
    print(list_loan[j])
    print(list_loan_total[j])
    print(list_loan[j]/list_loan_total[j])
    # print(list_loan[j]/list_loan_total[j]/ration)
'''

'''
print("contact:")
list_contact_total =[0,0]
list_contact =[0,0]
for i in range(0,len(test_client_list)):
    if test_client_list[i].contact == "cellular": list_contact_total[0] = list_contact_total[0] + 1
    if test_client_list[i].contact == "telephone": list_contact_total[1] = list_contact_total[1] + 1
    if test_client_list[i].Output == 1:
        if test_client_list[i].contact == "cellular": list_contact[0] = list_contact[0] + 1
        if test_client_list[i].contact == "telephone": list_contact[1] = list_contact[1] + 1
for j in range(0,len(list_contact_total)):
    # print(list_contact[j])
    # print(list_contact_total[j])
    # print(list_contact[j]/list_contact_total[j])
    print(list_contact[j]/list_contact_total[j]/ration)
'''

'''
print("month:")
list_month_total =[0,0,0,0,0,0,0,0,0,0]
list_month =[0,0,0,0,0,0,0,0,0,0]
for i in range(0,len(test_client_list)):
    if test_client_list[i].month == "may": list_month_total[0] = list_month_total[0] + 1
    if test_client_list[i].month == "nov": list_month_total[1] = list_month_total[1] + 1
    if test_client_list[i].month == "apr": list_month_total[2] = list_month_total[2] + 1
    if test_client_list[i].month == "aug": list_month_total[3] = list_month_total[3] + 1
    if test_client_list[i].month == "sep": list_month_total[4] = list_month_total[4] + 1
    if test_client_list[i].month == "jun": list_month_total[5] = list_month_total[5] + 1
    if test_client_list[i].month == "oct": list_month_total[6] = list_month_total[6] + 1
    if test_client_list[i].month == "jul": list_month_total[7] = list_month_total[7] + 1
    if test_client_list[i].month == "dec": list_month_total[8] = list_month_total[8] + 1
    if test_client_list[i].month == "mar": list_month_total[9] = list_month_total[9] + 1
    if test_client_list[i].Output == 1:
        if test_client_list[i].month == "may": list_month[0] = list_month[0] + 1
        if test_client_list[i].month == "nov": list_month[1] = list_month[1] + 1
        if test_client_list[i].month == "apr": list_month[2] = list_month[2] + 1
        if test_client_list[i].month == "aug": list_month[3] = list_month[3] + 1
        if test_client_list[i].month == "sep": list_month[4] = list_month[4] + 1
        if test_client_list[i].month == "jun": list_month[5] = list_month[5] + 1
        if test_client_list[i].month == "oct": list_month[6] = list_month[6] + 1
        if test_client_list[i].month == "jul": list_month[7] = list_month[7] + 1
        if test_client_list[i].month == "dec": list_month[8] = list_month[8] + 1
        if test_client_list[i].month == "mar": list_month[9] = list_month[9] + 1
for j in range(0,len(list_month_total)):
    # print(list_month[j])
    # print(list_month_total[j])
    # print(list_month[j]/list_month_total[j])
    print(list_month[j]/list_month_total[j]/ration)
'''
'''
print("day_of_week")
list_week_total =[0,0,0,0,0]
list_week =[0,0,0,0,0]
for i in range(0,len(test_client_list)):
    if test_client_list[i].day_of_week == "mon": list_week_total[0] = list_week_total[0] +1
    if test_client_list[i].day_of_week == "tue": list_week_total[1] = list_week_total[1] +1
    if test_client_list[i].day_of_week == "wed": list_week_total[2] = list_week_total[2] +1
    if test_client_list[i].day_of_week == "thu": list_week_total[3] = list_week_total[3] +1
    if test_client_list[i].day_of_week == "fri": list_week_total[4] = list_week_total[4] +1
    if test_client_list[i].Output == 1:
        if test_client_list[i].day_of_week == "mon": list_week[0] = list_week[0] + 1
        if test_client_list[i].day_of_week == "tue": list_week[1] = list_week[1] + 1
        if test_client_list[i].day_of_week == "wed": list_week[2] = list_week[2] + 1
        if test_client_list[i].day_of_week == "thu": list_week[3] = list_week[3] + 1
        if test_client_list[i].day_of_week == "fri": list_week[4] = list_week[4] + 1
for j in range(0,len(list_week_total)):
    # print(list_week[j])
    # print(list_week_total[j])
    # print(list_week[j]/list_week_total[j])
    print(list_week[j]/list_week_total[j]/ration)
'''

'''
print("duration:")
list_duration_total =[0,0,0,0,0,0,0,0,0,0,0,0,0]
list_duration =[0,0,0,0,0,0,0,0,0,0,0,0,0]
for i in range(0,len(test_client_list)):
    temp = test_client_list[i].duration
    if temp <= 60: list_duration_total[0] = list_duration_total[0] + 1;
    if temp > 60 and temp <= 100: list_duration_total[1] = list_duration_total[1] + 1;
    if temp > 100 and temp <= 150: list_duration_total[2] = list_duration_total[2] + 1;
    if temp > 150 and temp <= 200: list_duration_total[3] = list_duration_total[3] + 1;
    if temp > 200 and temp <= 250: list_duration_total[4] = list_duration_total[4] + 1;
    if temp > 250 and temp <= 300: list_duration_total[5] = list_duration_total[5] + 1;
    if temp > 300 and temp <= 400: list_duration_total[6] = list_duration_total[6] + 1;
    if temp > 400 and temp <= 450: list_duration_total[7] = list_duration_total[7] + 1;
    if temp > 450 and temp <= 500: list_duration_total[8] = list_duration_total[8] + 1;
    if temp > 500 and temp <= 550: list_duration_total[9] = list_duration_total[9] + 1;
    if temp > 550 and temp <= 600: list_duration_total[10] = list_duration_total[10] + 1;
    if temp > 600 and temp <= 800: list_duration_total[11] = list_duration_total[11] + 1;
    if temp > 800: list_duration_total[12] = list_duration_total[12] + 1;
    if test_client_list[i].Output == 1:
        if temp <= 60: list_duration[0] = list_duration[0] + 1;
        if temp > 60 and temp <= 100: list_duration[1] = list_duration[1] + 1;
        if temp > 100 and temp <= 150: list_duration[2] = list_duration[2] + 1;
        if temp > 150 and temp <= 200: list_duration[3] = list_duration[3] + 1;
        if temp > 200 and temp <= 250: list_duration[4] = list_duration[4] + 1;
        if temp > 250 and temp <= 300: list_duration[5] = list_duration[5] + 1;
        if temp > 300 and temp <= 400: list_duration[6] = list_duration[6] + 1;
        if temp > 400 and temp <= 450: list_duration[7] = list_duration[7] + 1;
        if temp > 450 and temp <= 500: list_duration[8] = list_duration[8] + 1;
        if temp > 500 and temp <= 550: list_duration[9] = list_duration[9] + 1;
        if temp > 550 and temp <= 600: list_duration[10] = list_duration[10] + 1;
        if temp > 600 and temp <= 800: list_duration[11] = list_duration[11] + 1;
        if temp > 800: list_duration[12] = list_duration[12] + 1;

for j in range(0,len(list_duration)):
    # print(list_duration[j])
    # print(list_duration_total[j])
    # print(list_duration[j]/list_duration_total[j])
    print(list_duration[j]/list_duration_total[j]/ration)
'''

'''
print("campaign")
list_campaign_total =[0,0,0,0,0]
list_campaign =[0,0,0,0,0]
for i in range(0,len(test_client_list)):
    if test_client_list[i].campaign <= 10: list_campaign_total[0] = list_campaign_total[0] + 1;
    if test_client_list[i].campaign > 10 and test_client_list[i].campaign <= 20: list_campaign_total[1] = \
        list_campaign_total[1] + 1;
    if test_client_list[i].campaign > 20 and test_client_list[i].campaign <= 30: list_campaign_total[2] = \
        list_campaign_total[2] + 1;
    if test_client_list[i].campaign > 30 and test_client_list[i].campaign <= 40: list_campaign_total[3] = \
        list_campaign_total[3] + 1;
    if test_client_list[i].campaign > 50: list_campaign_total[4] = list_campaign_total[4] + 1;
    if test_client_list[i].Output == 1:
        if test_client_list[i].campaign <= 10: list_campaign[0] = list_campaign[0] + 1;
        if test_client_list[i].campaign > 10 and test_client_list[i].campaign <= 20: list_campaign[1] = \
            list_campaign[1] + 1;
        if test_client_list[i].campaign > 20 and test_client_list[i].campaign <= 30: list_campaign[2] = \
            list_campaign[2] + 1;
        if test_client_list[i].campaign > 30 and test_client_list[i].campaign <= 40: list_campaign[3] = \
            list_campaign[3] + 1;
        if test_client_list[i].campaign > 50: list_campaign[4] = list_campaign[4] + 1;
for j in range(0,len(list_campaign)):
    # print(list_duration[j])
    # print(list_duration_total[j])
    # print(list_duration[j]/list_duration_total[j])
    print(list_campaign[j]/list_campaign_total[j]/ration)
'''
'''
print("pdays:")
list_pdays_total =[0,0,0,0]
list_pdays =[0,0,0,0]
for i in range(0,len(test_client_list)):
    if test_client_list[i].pdays <= 8: list_pdays_total[0] = list_pdays_total[0] + 1;
    if test_client_list[i].pdays > 8 and test_client_list[i].pdays <= 16: list_pdays_total[1] = \
        list_pdays_total[1] + 1;
    if test_client_list[i].pdays > 16 and test_client_list[i].pdays <= 30: list_pdays_total[2] = \
        list_pdays_total[2] + 1;
    if test_client_list[i].pdays == 999: list_pdays_total[3] = list_pdays_total[3] + 1;
    if test_client_list[i].Output == 1:
        if test_client_list[i].pdays <= 8: list_pdays[0] = list_pdays[0] + 1;
        if test_client_list[i].pdays > 8 and test_client_list[i].pdays <= 16: list_pdays[1] = \
            list_pdays[1] + 1;
        if test_client_list[i].pdays > 16 and test_client_list[i].pdays <= 30: list_pdays[2] = \
            list_pdays[2] + 1;
        if test_client_list[i].pdays == 999: list_pdays[3] = list_pdays[3] + 1;
for j in range(0,len(list_pdays)):
    # print(list_pdays[j])
    # print(list_pdays_total[j])
    # print(list_pdays[j]/list_pdays_total[j])
    print(list_pdays[j]/list_pdays_total[j]/ration)
'''
'''
print("previous:")
list_previous_total =[0,0,0,0,0,0,0,0]
list_previous =[0,0,0,0,0,0,0,0]
for i in range(0,len(test_client_list)):
    if test_client_list[i].previous == 0: list_previous_total[0] = list_previous_total[0] + 1
    if test_client_list[i].previous == 1: list_previous_total[1] = list_previous_total[1] + 1
    if test_client_list[i].previous == 2: list_previous_total[2] = list_previous_total[2] + 1
    if test_client_list[i].previous == 3: list_previous_total[3] = list_previous_total[3] + 1
    if test_client_list[i].previous == 4: list_previous_total[4] = list_previous_total[4] + 1
    if test_client_list[i].previous == 5: list_previous_total[5] = list_previous_total[5] + 1
    if test_client_list[i].previous == 6: list_previous_total[6] = list_previous_total[6] + 1
    if test_client_list[i].previous == 7: list_previous_total[7] = list_previous_total[7] + 1
    if test_client_list[i].Output == 1:
        if test_client_list[i].previous == 0: list_previous[0] = list_previous[0] + 1
        if test_client_list[i].previous == 1: list_previous[1] = list_previous[1] + 1
        if test_client_list[i].previous == 2: list_previous[2] = list_previous[2] + 1
        if test_client_list[i].previous == 3: list_previous[3] = list_previous[3] + 1
        if test_client_list[i].previous == 4: list_previous[4] = list_previous[4] + 1
        if test_client_list[i].previous == 5: list_previous[5] = list_previous[5] + 1
        if test_client_list[i].previous == 6: list_previous[6] = list_previous[6] + 1
        if test_client_list[i].previous == 7: list_previous[7] = list_previous[7] + 1
for j in range(0,len(list_previous_total)):
    # print(list_previous[j])
    # print(list_previous_total[j])
    # print(list_previous[j]/list_previous_total[j])
    print(list_previous[j]/list_previous_total[j]/ration)
'''
'''
print("poutcome:")
list_previous_total =[0,0,0]
list_previous =[0,0,0]
for i in range(0,len(test_client_list)):
    if test_client_list[i].poutcome == "failure": list_previous_total[0] = list_previous_total[0] + 1
    if test_client_list[i].poutcome == "nonexistent": list_previous_total[1] = list_previous_total[1] + 1
    if test_client_list[i].poutcome == "success": list_previous_total[2] = list_previous_total[2] + 1
    if test_client_list[i].Output == 1:
        if test_client_list[i].poutcome == "failure": list_previous[0] = list_previous[0] +1
        if test_client_list[i].poutcome == "nonexistent": list_previous[1] = list_previous[1] +1
        if test_client_list[i].poutcome == "success": list_previous[2] = list_previous[2] +1
for j in range(0,len(list_previous_total)):
    # print(list_previous[j])
    # print(list_previous_total[j])
    # print(list_previous[j]/list_previous_total[j])
    print(list_previous[j]/list_previous_total[j]/ration)
'''
'''
print("emp:")
list_emp_total =[0,0,0] # -4  --- 2
list_emp =[0,0,0]
for i in range(0,len(test_client_list)):
    if test_client_list[i].emp_var_rate <= -2:list_emp_total[0] = list_emp_total[0] +1
    if test_client_list[i].emp_var_rate > -2 and test_client_list[i].emp_var_rate <= 0: list_emp_total[1] = list_emp_total[1] + 1
    if test_client_list[i].emp_var_rate > -0 and test_client_list[i].emp_var_rate <= 2: list_emp_total[2] = list_emp_total[2] + 1
    if test_client_list[i].Output == 1:
        if test_client_list[i].emp_var_rate <= -2: list_emp[0] = list_emp[0] + 1
        if test_client_list[i].emp_var_rate > -2 and test_client_list[i].emp_var_rate <= 0: list_emp[1] = \
                                                                                            list_emp[1] + 1
        if test_client_list[i].emp_var_rate > -0 and test_client_list[i].emp_var_rate <= 2: list_emp[2] = \
                                                                                            list_emp[2] + 1
for j in range(0,len(list_emp_total)):
    # print(list_emp[j])
    # print(list_emp_total[j])
    # print(list_emp[j]/list_emp_total[j])
    print(list_emp[j]/list_emp_total[j]/ration)
'''
'''
print("price:")
list_price_total =[0,0,0]
list_price =[0,0,0]
for i in range(0,len(test_client_list)):
    temp = test_client_list[i].cons_price_idx
    if temp < 93: list_price_total[0] = list_price_total[0] + 1
    if temp >= 93 and temp < 94: list_price_total[1] = list_price_total[1] + 1
    if temp >= 94: list_price_total[2] = list_price_total[2] + 1
    if test_client_list[i].Output == 1:
        if temp < 93:list_price[0] = list_price[0] + 1
        if temp >= 93 and temp < 94: list_price[1] = list_price[1] + 1
        if temp >= 94:list_price[2] = list_price[2] + 1
for j in range(0,len(list_price_total)):
    # print(list_price[j])
    # print(list_price_total[j])
    # print(list_price[j]/list_price_total[j])
    print(list_price[j]/list_price_total[j]/ration)
'''
'''
print("conf:")
list_conf_total =[0,0,0] # -51 === -25
list_conf =[0,0,0]
for i in range(0,len(test_client_list)):
    temp = test_client_list[i].cons_conf_idx
    if temp < -43: list_conf_total[0] = list_conf_total[0] + 1
    if temp >= -43 and temp < -34: list_conf_total[1] = list_conf_total[1] + 1
    if temp >= -34: list_conf_total[2] = list_conf_total[2] + 1
    if test_client_list[i].Output == 1:
        if temp < -43: list_conf[0] = list_conf[0] + 1
        if temp >= -43 and temp < -34: list_conf[1] = list_conf[1] + 1
        if temp >= -34: list_conf[2] = list_conf[2] + 1
for j in range(0,len(list_conf_total)):
    # print(list_conf[j])
    # print(list_conf_total[j])
    # print(list_conf[j]/list_conf_total[j])
    print(list_conf[j]/list_conf_total[j]/ration)
'''
'''
print("euribor3m")
list_euribor3m_total =[0,0,0,0]
list_euribor3m =[0,0,0,0]
for i in range(0,len(test_client_list)):
    temp = test_client_list[i].euribor3m
    if temp < 1: list_euribor3m_total[0] = list_euribor3m_total[0] + 1
    if temp >= 1 and temp < 2: list_euribor3m_total[1] = list_euribor3m_total[1] + 1
    if temp >= 2 and temp < 4: list_euribor3m_total[2] = list_euribor3m_total[2] + 1
    if temp >= 4: list_euribor3m_total[3] = list_euribor3m_total[3] + 1
    if test_client_list[i].Output == 1:
        if temp < 1: list_euribor3m[0] = list_euribor3m[0] + 1
        if temp >= 1 and temp < 2: list_euribor3m[1] = list_euribor3m[1] + 1
        if temp >= 2 and temp < 4: list_euribor3m[2] = list_euribor3m[2] + 1
        if temp >= 4: list_euribor3m[3] = list_euribor3m[3] + 1
for j in range(0,len(list_euribor3m_total)):
    # print(list_euribor3m[j])
    # print(list_euribor3m_total[j])
    # print(list_euribor3m[j]/list_euribor3m_total[j])
    print(list_euribor3m[j]/list_euribor3m_total[j]/ration)
'''

'''
print("employed")
list_employed_total =[0,0,0]
list_employed =[0,0,0]
for i in range(0,len(test_client_list)):
    temp = test_client_list[i].nr_employed # 4963 -5229
    if temp < 5052: list_employed_total[0] = list_employed_total[0] + 1
    if temp >= 5052 and temp < 5141: list_employed_total[1] = list_employed_total[1] + 1
    if temp >= 5141: list_employed_total[2] = list_employed_total[2] + 1
    if test_client_list[i].Output == 1:
        if temp < 5052: list_employed[0] = list_employed[0] + 1
        if temp >= 5052 and temp < 5141: list_employed[1] = list_employed[1] + 1
        if temp >= 5141: list_employed[2] = list_employed[2] + 1
for j in range(0,len(list_employed_total)):
    # print(list_employed[j])
    # print(list_employed_total[j])
    # print(list_employed[j]/list_employed_total[j])
    print(list_employed[j]/list_employed_total[j]/ration)

'''

'''
count = 0;
count_1 = 0;
print("student-success")
for i in range(0,len(test_client_list)):
    print(test_client_list[i].job,test_client_list[i].previous)
    if test_client_list[i].job == "student" and test_client_list[i].previous == 6:
        count = count + 1
        if test_client_list[i].Output == 1:
            count_1 = count_1 + 1
print(count_1,count)
print(count_1/count/ration)
'''




