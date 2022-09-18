#血流音特徵，音檔切音框(音框0.1秒，overlap=0.05)
import numpy as np
import librosa
from scipy.fft import fft

def ACF(frame):
    flen = len(frame)
    acf = np.zeros(flen)
    for i in range(flen):
        acf[i] = np.sum(frame[i:flen]*frame[0:flen-i])
    return acf
def Save(value):
    save.append(value)
    return save
def invert(list1):
    return [row[::-1] for row in list1]

label = "0" #0:block； 1:normal  這裡設定CSV的target欄填甚麼
four = 100
three = (four/4)*3
two = (four/4)*2
one = four/4

path = "C:\\Users\\u1070\\OneDrive\\桌面\\2020-10-12-11am\\06366775-1-2020-10-12-11-12-07"
fw, sr = librosa.load(path+".wav", sr=44100)
save =[]

'''
#音框設定 (詳細解說參考:https://reurl.cc/ARM4zQ)
音框率:每秒鐘切幾個音框，算法: 取樣頻率(sr)/音框長度(frameSize)-重疊點數(overlap) =44100/(4410-2205)=20

sr:每秒產生幾個點數
frameSize:每個音框幾個點數
音框大小(秒數)=frameSize/sr
'''
#time = np.arange(0, len(fw)) * (1.0 / sr)   #音檔總時長
frameSize = 4410    #音框長度
overlap = 2205      #重疊點數

#擷取pitch
for i in range(0, (len(fw)-frameSize),(frameSize-overlap)):
    index1 = i
    index2 = (i+frameSize)
    acf = ACF(fw[index1:index2])
    acf_max = acf.max()
    value = sr/acf_max
    pitch = Save(value)

pitch = np.asarray(pitch)
pitch_T = np.array(pitch).reshape(len(pitch),1)
print("-pitch-")
#需要刪除的話，請貼這邊: pitch_T = np.delete(pitch_T, -1, axis=0)

print(pitch_T.shape)

#擷取MFCC
n = 12
mfcc = librosa.feature.mfcc(fw, sr=sr, n_mfcc=n, n_fft=frameSize, dct_type=2, hop_length=(frameSize-overlap))
print("-mfcc-")

colume = int(mfcc.shape[0])
row = int(mfcc.shape[1])
mfcc_T = np.array(mfcc).reshape(row,colume)
mfcc_T = np.delete(mfcc_T, -1, axis=0)
mfcc_T = np.delete(mfcc_T, -1, axis=0)
mfcc_T = np.delete(mfcc_T, -1, axis=0)
#需要刪除的話，請貼這邊: mfcc_T = np.delete(mfcc_T, -1, axis=0)

print(mfcc_T.shape)


#擷取zero rate
zero_rate = librosa.feature.zero_crossing_rate(fw, frame_length=frameSize, hop_length=(frameSize-overlap))
print("-zero_rate-") #

colume = int(zero_rate.shape[0])
row = int(zero_rate.shape[1])
zero_rate_T = np.array(zero_rate).reshape(row,colume)
zero_rate_T = np.delete(zero_rate_T, -1, axis=0)
zero_rate_T = np.delete(zero_rate_T, -1, axis=0)
zero_rate_T = np.delete(zero_rate_T, -1, axis=0)
#需要刪除的話，請貼這邊: zero_rate_T = np.delete(zero_rate_T, -1, axis=0)

print(zero_rate_T.shape)

#寫標籤
colume = int(zero_rate_T.shape[0])
row = int(zero_rate_T.shape[1])
target = np.full([colume, row], label)
print("-target-")
print(target.shape)


# 擷取射頻功率比 RFpower
def split_frame(sr, fw, split_len=frameSize, overlap=(1 - (overlap / frameSize))):
    frames = []

    for i in range(0, len(fw)):
        start_idx = i * int((split_len * overlap))
        end_idx = start_idx + split_len
        # print(start_idx,'~',end_idx)
        if end_idx >= len(fw):
            break

        frame = fw[start_idx:end_idx]
        frames.append(np.array(frame))

    return frames

frames = split_frame(sr, fw)
pr1 = []
pr2 = []
pr3 = []
pr4 = []

part1 = []
part2 = []
part3 = []
part4 = []

for j in range(0, len(frames)):
    ft = fft(frames[j])
    magnitude = np.absolute(ft.imag)  # 虛部
    frequency = ft.real  # 實部

    for k in range(0, len(frequency)):
        if frequency[k] <= one:
            pr1.append(magnitude[k])

        elif (frequency[k] > one) & (frequency[k] <= two):
            pr2.append(magnitude[k])

        elif (frequency[k] > two) & (frequency[k] <= three):
            pr3.append(magnitude[k])

        elif (frequency[k] > three) & (frequency[k] <= four):
            pr4.append(magnitude[k])

    #各頻段總功率
    p1 = np.sum(pr1)
    p2 = np.sum(pr2)
    p3 = np.sum(pr3)
    p4 = np.sum(pr4)

    #總功率
    ptotal = p1 + p2 + p3 + p4

    #各頻段射功率比
    r1 = p1 / ptotal
    r2 = p2 / ptotal
    r3 = p3 / ptotal
    r4 = p4 / ptotal

    #紀錄每筆音框樣本的射頻功率比
    part1.append(r1)
    part2.append(r2)
    part3.append(r3)
    part4.append(r4)

part1 = np.array(part1)
pr1_T = part1.reshape((len(part1), 1))
print("-pr1-")
#需要刪除的話，請貼這邊: pr1_T = np.delete(pr1_T, -1, axis=0)

print(pr1_T.shape)

part2 = np.array(part2)
pr2_T = part2.reshape((len(part2), 1))
print("-pr2-")
#需要刪除的話，請貼這邊: pr2_T = np.delete(pr2_T, -1, axis=0)

print(pr2_T.shape)

part3 = np.array(part3)
pr3_T = part3.reshape((len(part3), 1))
print("-pr3-")
#需要刪除的話，請貼這邊: pr3_T = np.delete(pr3_T, -1, axis=0)

print(pr3_T.shape)

part4 = np.array(part4)
pr4_T = part4.reshape((len(part4), 1))
print("-pr4-")
#需要刪除的話，請貼這邊: pr4_T = np.delete(pr4_T, -1, axis=0)

print(pr4_T.shape)

#寫入CSV
new = np.hstack([pitch_T, mfcc_T, zero_rate_T, pr1_T, pr2_T, pr3_T, pr4_T, target]) #
four_num = str(four)
df1 = np.savetxt(path+"_"+label+"_"+four_num+"_FeatureData-Flow.csv", new, fmt="%s", delimiter=",",
                 header="pitch,mfcc1,mfcc2,mfcc3,mfcc4,mfcc5,mfcc6,mfcc7,mfcc8,mfcc9,mfcc10,mfcc11,mfcc12,zero_cross,p1,p2,p3,p4,target") #