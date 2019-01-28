from fuzzywuzzy import fuzz
from fuzzywuzzy import process

import pandas as pd
import csv
import sys
import xlrd
from openpyxl import Workbook


# from operator import itemgetter
#
# try:
#     arg1, arg2, arg3, arg4=sys.argv
#
# except:
#     print("e.g. Input format: python match.py df1.csv df2.csv out.csv")
#     exit()



def LCSubStr(X, Y, m, n): #calculate longest common string

    LCSuff = [[0 for k in range(n + 1)] for l in range(m + 1)]

    # longest common substring
    result = 0

    for i in range(m + 1):
        for j in range(n + 1):
            if (i == 0 or j == 0):
                LCSuff[i][j] = 0
            else:
                LCSuff[i][j] = max(LCSuff[i][j], LCSuff[i - 1][j])
                LCSuff[i][j] = max(LCSuff[i][j], LCSuff[i][j - 1])
                if (X[i-1]==Y[j-1]):
                    LCSuff[i][j] = max(LCSuff[i][j], LCSuff[i - 1][j - 1] + 1)
                    result = max(result, LCSuff[i][j])



    return result

def cmp_f1(df01, df02): #calculate match
    aa=df01.split(" ")
    df01_l=aa[0]   #last name
    df01_f=aa[1]

    bb=df02.split(" ")
    df02_l=bb[0]
    df02_f=bb[1]

    l01=len(df01_l)
    l02=len(df02_l)

    ll=min(l01, l02)

    lc=LCSubStr(df01_l, df02_l, l01, l02)

    if lc>=ll/2:
        l01 = len(df01_f)
        l02 = len(df02_f)

        ll = min(l01, l02)

        lc = LCSubStr(df01_f, df02_f, l01, l02)

        if lc>=ll/2:
            return 1

    return 0



reader1 = csv.reader(open('df3.csv', 'r'), delimiter=',', quotechar='"')#read first csv
wb1=Workbook()
ws1=wb1.active

r1=next(reader1)

flag=0
name01=[]
no=-1
ele=["a", 0, 0] #initialize
name01.append(ele)
ln=0
# aa=["aa", 0, 0]
for rr1 in reader1:
    no=no+1
    ws1.append(rr1)

    if flag==0:
        ele[0]=rr1[4]
        ele[1]=no
        flag=1
    else:
        if ele[0]==rr1[4]:
            continue
        else:
            ele[2]=no

            ch_name=ele[0].split(" ")
            chn=""
            for ci in range(1, len(ch_name)):
                chn=chn+ch_name[ci]

            name01[ln]=[chn+" "+ch_name[0], ele[1], no]#save F1_Name, start and end position of the name

            name01.append(ele)
            ln=ln+1

            ele[0]=rr1[4]
            ele[1]=no


ele[2]=no+1
ch_name=ele[0].split(" ")
chn=""
for ci in range(1, len(ch_name)):
    chn=chn+ch_name[ci]

name01[ln]=[chn+" "+ch_name[0], ele[1], no+1]
name01.sort()#sort by name

ln01=ln+1

wb1.save("ex01.xlsx")





reader2 = csv.reader(open('df2.csv', 'r'), delimiter=',', quotechar='"')
wb2=Workbook()
ws2=wb2.active

r2=next(reader2)

flag=0
name02=[]
no=-1
ele=["a", 0, 0] #initialize
name02.append(ele)
ln=0
# aa=["aa", 0, 0]
for rr2 in reader2:
    no=no+1
    ws2.append(rr2)

    if flag==0:
        ele[0]=rr2[4]
        ele[1]=no
        flag=1
    else:
        if ele[0]==rr2[4]:
            continue
        else:
            ele[2]=no

            ch_name = ele[0].split(" ")
            chn = ""
            for ci in range(1, len(ch_name)):
                chn = chn + ch_name[ci]

            name02[ln]=[chn+" "+ch_name[0], ele[1], no]

            name02.append(ele)
            ln=ln+1


            ele[0]=rr2[4]
            ele[1]=no


ele[2]=no+1

ch_name=ele[0].split(" ")
chn=""
for ci in range(1, len(ch_name)):
    chn=chn+ch_name[ci]
name02[ln]=[chn+" "+ch_name[0], ele[1], no+1]

name02.sort()
ln02=ln

wb2.save("ex02.xlsx")

rn01=0
rn02=0

with open("result.csv", "w", newline="") as f:
    writer=csv.writer(f)
    writer.writerow(["F1_ID", "Match_ID", "Exact_ID", "Close_ID", "No_Match_DF1", "No_Match_DF2", "Name_Match", "Dates_Match", "F2_Match"])

    loc01="ex01.xlsx"
    rs01=xlrd.open_workbook(loc01)
    sheet01=rs01.sheet_by_index(0)


    loc02="ex02.xlsx"
    rs02=xlrd.open_workbook(loc02)
    sheet02=rs02.sheet_by_index(0)


    while rn01<ln01 and rn02<ln02:


        cmpN=cmp_f1(name01[rn01][0], name02[rn02][0])
        if cmpN==1: # and fuzz.token_set_ratio(sheet01.cell_value(name01[rn01][1], 4), sheet02.cell_value(name02[rn02][1], 4))>=90:
            list1=[]
            list2=[]

            ad01=[]
            ad02=[]

            for i in range(name01[rn01][1], name01[rn01][2]):

                ad01=[]
                for j in range(7):
                    ad01.append(sheet01.cell_value(i, j+1))

                list1.append(ad01)

            while rn01+1<ln01:
                if name01[rn01][0]==name01[rn01+1][0]:
                    rn01=rn01+1
                    for i in range(name01[rn01][1], name01[rn01][2]):

                        ad01 = []
                        for j in range(7):
                            ad01.append(sheet01.cell_value(i, j + 1))

                        list1.append(ad01)
                else:
                    break

            for i in range(name02[rn02][1], name02[rn02][2]):

                ad02=[]
                for j in range(7):
                    ad02.append(sheet02.cell_value(i, j+1))

                list2.append(ad02)

            while rn02+1<ln02:
                if name02[rn02][0]==name02[rn02+1][0]:
                    rn02=rn02+1
                    for i in range(name02[rn02][1], name02[rn02][2]):

                        ad02 = []
                        for j in range(7):
                            ad02.append(sheet02.cell_value(i, j + 1))

                        list2.append(ad02)
                else:
                    break

            # F1_ID=list1[0][5]
            # Match_ID=list2[0][5]
            len_list1=len(list1)
            len_list2=len(list2)

            Exact_Match=""
            Close_Match=""
            No_Match_DF2=""
            No_Match_DF1 = ""
            F01=""
            F02=""

            multi_match=[]
            cnt_m=0

            match1=[]
            for j in range(len_list1):
                match1.append(0)

            for i in range(len_list2):
                no_match=1
                for j in range(len_list1):
                    if list2[i][0]==list1[j][0]:#compare Event_Date
                        no_match=0
                        match1[j]=1
                        if list2[i][1]==list1[j][1]:#If Event_Name is same, Exact_match
                            # Exact_Match=Exact_Match+" "+list2[i][4]+" "+list2[i][1]+" "+list2[i][0]+"\n"
                            # F01=F01+" "+list1[j][4]
                            # F02=F02+" "+list2[i][4]
                            ff=0
                            for ic in range(cnt_m):
                                if multi_match[ic][0]==list1[j][5] and multi_match[ic][1]==list2[i][5]:
                                    multi_match[ic][2]=multi_match[ic][2]+list2[i][4]+" "+list2[i][1]+" "+list2[i][0]+"\n"
                                    multi_match[ic][8]=multi_match[ic][8]+" "+list1[j][4]
                                    multi_match[ic][9] = multi_match[ic][9] + " " + list2[i][4]
                                    ff=1
                                    break

                            if ff==0:
                                multi_match.append([list1[j][5], list2[i][5], list2[i][4]+" "+list2[i][1]+" "+list2[i][0]+"\n", \
                                                    "", "", "", fuzz.token_set_ratio(list1[j][3], list2[i][3])/100, 1, list1[j][4], list2[i][4]])
                                cnt_m=cnt_m+1
                            break
                        else:#If Event_Name is not same, Close_match
                            # Close_Match=Close_Match+" "+list2[i][4]+" "+list2[i][1]+" "+list2[i][0]+"\n"
                            # F01 = F01 + " " + list1[j][4]
                            # F02 = F02 + " " + list2[i][4]
                            ff = 0
                            for ic in range(cnt_m):
                                if multi_match[ic][0] == list1[j][5] and multi_match[ic][1] == list2[i][5]:
                                    multi_match[ic][3] = multi_match[ic][2] + list2[i][4] + " " + list2[i][1] + " " + \
                                                         list2[i][0] + "\n"
                                    multi_match[ic][8] = multi_match[ic][8] + " " + list1[j][4]
                                    multi_match[ic][9] = multi_match[ic][9] + " " + list2[i][4]
                                    ff = 1
                                    break

                            if ff == 0:
                                multi_match.append([list1[j][5], list2[i][5],
                                                    "", list2[i][4] + " " + list2[i][1] + " " + list2[i][0] + "\n", \
                                                     "", "", fuzz.token_set_ratio(list1[j][3], list2[i][3]) / 100, 1,\
                                                    list1[j][4], list2[i][4]])
                                cnt_m = cnt_m + 1
                            break

                if no_match==1:#If there is no Match, No_Match

                    for ci in range(cnt_m):
                        if multi_match[ci][1]==list2[i][5]:
                            multi_match[ci][5]=multi_match[ci][5]+list2[i][4]+" "+list2[i][1]+" "+list2[i][0]+"\n"
                            break
                    # No_Match_DF2=No_Match_DF2+" "+list2[i][4]+" "+list2[i][1]+" "+list2[i][0]+"\n"

            for j in range(len_list1):  # find no match in csv01
                if match1[j]==0:
                    for ci in range(cnt_m):
                        if multi_match[ci][1]==list1[j][5]:
                            multi_match[ci][4]=multi_match[ci][5]+list1[j][4]+" "+list1[j][1]+" "+list1[j][0]+"\n"
                            break
                    # No_Match_DF1=No_Match_DF1+" "+list1[j][4]+" "+list1[j][1]+" "+list1[j][0]+"\n"

            # Name_Match=fuzz.token_set_ratio(list1[0][3], list2[0][3])/100
            # F2_Match=fuzz.token_set_ratio(F01, F02)/100

            for ci in range(cnt_m):
                writer.writerow([multi_match[ci][0], multi_match[ci][1], multi_match[ci][2], multi_match[ci][3],\
                                 multi_match[ci][4], multi_match[ci][5], multi_match[ci][6], multi_match[ci][7],\
                                 fuzz.token_set_ratio(multi_match[ci][8], multi_match[ci][9])/100])

            # writer.writerow([F1_ID, Match_ID, Exact_Match, Close_Match, No_Match_DF1, No_Match_DF2, Name_Match, 1, F2_Match])


            rn01=rn01+1
            rn02=rn02+1

        else:
            if name01[rn01][0]<name02[rn02][0]:
                rn01=rn01+1
            else:
                rn02=rn02+1







