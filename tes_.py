from ctypes import sizeof
import os
import pandas as pd 
import xlrd
import random
random.seed(1)
def readDir(dirPath):
    allFiles = []
    if os.path.isdir(dirPath):
        fileList = os.listdir(dirPath)
        for f in fileList:
            ff=f
            f = dirPath+'/'+f
            
            if os.path.isdir(f):
                subFiles = readDir(f)
                allFiles = subFiles + allFiles
            else:
                if f!='./data/hsk_article/.DS_Store':
                    allFiles.append(ff)
        return allFiles
    else:
        return 'Error,not a dir'
name_list=readDir("./data/hsk_article")
print(len(name_list))
def getNumList(start, end, n):
    numsArray = set()
    while len(numsArray) < n:
        numsArray.add(random.randint(start, end))        
    return list(numsArray)
print(len(name_list))
arlis=[]
for i in range(len(name_list)):
    f=open('./data/hsk_article/'+name_list[i],'r',encoding='gb18030')
    content=f.readlines()
    ss=''.join(content)
    ss=ss.strip()
    ss=ss.strip(' ()（）') 
    ss=ss.replace("P}",'')
    ss=ss.strip("（")
    arlis.append(ss)
xy = pd.DataFrame([
    pd.Series(name_list,name='文件名'),
    pd.Series(arlis,name='文章内容'),
]).T
df1=pd.read_excel('./data/hsk_rank.xlsx')
print(df1.info())
xyz=pd.merge(xy,df1,on="文件名",how='inner')
print(xyz)
xyz.to_csv("./data/result.csv",index=False)
num_list=getNumList(0,11146,11147)
x1=num_list[0:8917]
x2=num_list[8917:10032]
x3=num_list[10032:]
xyz1=xyz.take(x1)
xyz2=xyz.take(x2)
xyz3=xyz.take(x3)
xyz1.to_csv("./data/train.csv",index=False)
xyz2.to_csv("./data/value.csv",index=False)
xyz3.to_csv("./data/test.csv",index=False)