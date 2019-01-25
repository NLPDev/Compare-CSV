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



def LCSubStr(X, Y, m, n):

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

reader1 = csv.reader(open('df1.csv', 'r'), delimiter=',', quotechar='"')
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
            name01[ln]=[ele[0], ele[1], no]

            name01.append(ele)
            ln=ln+1


            ele[0]=rr1[4]
            ele[1]=no


ele[2]=no+1
name01[ln]=ele
name01.sort()

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
            name02[ln]=[ele[0], ele[1], no]

            name02.append(ele)
            ln=ln+1


            ele[0]=rr2[4]
            ele[1]=no


ele[2]=no+1
name02[ln]=ele
name02.sort()


wb2.save("ex02.xlsx")