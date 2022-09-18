import os
import pandas as pd
# 指定要查詢的路徑
label = "0"
pathName = "C:\\Users\\wxy\\PycharmProjects\\project108\\block"
# 列出指定路徑底下所有檔案(包含資料夾)
allFileList = os.listdir(pathName)
# 逐一查詢檔案清單
dataset = pd.DataFrame()
for i in allFileList:
    filePath = pathName+'\\'+i
    print(filePath)
    readfile = pd.read_csv((filePath))
    print(readfile.shape)
    dataset = dataset.append(readfile)
    print("→",dataset.shape)
print("=======================================================================")
print(dataset)
print("total:",dataset.shape)
dataset.to_csv(pathName+'\\'+label+"_dataSet.csv", index=False)