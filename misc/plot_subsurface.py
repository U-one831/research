import os
from PIL import Image
import numpy as np
import pdb
import pickle
import matplotlib.pylab as plt
import cv2
import pandas as pd
import glob
import copy

# setting layer
layer = 14 #出力したい階層の値を代入
layerNum = 23 #全層数
if layer == 1:
    print("1層目の上面はなし")
    sys.exit()
#-------------------

#-------------------
# plot observation sites
# 各pickleデータのパス
dataName = "Nankai_JIVSM111110-50m_non1st_100km_130814.csv"
dirPath = "nankaiUnderGroundData" # データの保存先
picklePath = dirPath+os.sep+"underGroundCSV2Dict.pickle"

if not os.path.isdir(dirPath):
    os.makedirs(dirPath)

# csv を pickleに変換したデータがあれば読み込み・なければ作成
if os.path.isfile(picklePath):
    pass
else:
    csvData = csv.reader(open(dataName, "r"), delimiter=",", doublequote=True, lineterminator="\r\n")

    # 42行目までは層の上面深さに関係ない
    for _ in range(42):
        _ = next(csvData)

    # 保存先リスト
    XPO,YPO,RX,RY = [],[],[],[] # 座標
    JLON,JLAT,WLON,WLAT = [],[],[],[] # 緯度経度（日本基準と世界基準）
    SEL,NL = [],[] # 謎のパラメータ
    Ps = [[] for _ in range(layerNum)] # 層の番号
    Ts = [[] for _ in range(layerNum)] # 層の厚さ

    ite = 0
    for row in csvData:
        ite += 1
        print("\r csv2pickle : already read {} rows".format(ite),end="")
        XPO.append(int(row[1]))
        YPO.append(int(row[0]))
        JLON.append(float(row[2]))
        JLAT.append(float(row[3]))
        WLON.append(float(row[4]))
        WLAT.append(float(row[5]))
        RX.append(int(row[6]))
        RY.append(int(row[7]))
        SEL.append(row[8])
        NL.append(row[9])
        for i in range(layerNum):
            Ps[i].append(float(row[10+2*i]))
            Ts[i].append(float(row[11+2*i]))

    dataDict = {"XPO":XPO,"YPO":YPO,"JLON":JLON,"JLAT":JLAT,"WLON":WLON,"WLAT":WLAT,"RX":RX,"RY":RY,"SEL":SEL,"NL":NL,"Ps":Ps,"Ts":Ts}

    pickle.dump(dataDict,open(picklePath,"wb"))
    print("saved")

#色分けの設定
# 深さの色分けをする際の閾値
threlist = [0,50,100,200,300,400,500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,8000,10000]

def thresh(x):
    def judge(x,sta,end):
        return (x>sta) & (x<=end)

    dRGB = int(256/(len(threlist)-1))

    x[x<threlist[0]] = 0    
    for i in range(len(threlist)-1):
        x[judge(x,threlist[i],threlist[i+1])] = dRGB*(i+1)
    x[x>threlist[-1]] = dRGB*(len(threlist)-1)

    return x
#---
def cmap(x): # xは0~255の範囲
    # pdb.set_trace()
    hsvcmap = plt.get_cmap("hsv")
    x = x.astype("int")
    img = (x+210)%256
    return hsvcmap(img)
#--- 
def colorBar(n=256,width=20): # n:サンプル数 width:幅
    x = np.array([[i for _ in range(width)] for i in range(n)]) # 0~1 の範囲
    return cmap(x)
#---
def bw2color(img):
    # pdb.set_trace()
    colorImg = cmap(thresh(img))[:,:,:3]
    colorImg[existImg==0] = 1
    return colorImg
#---

sea_path = '../data/sea.png'
sea = np.array(Image.open(sea_path))/255
files = sorted(glob.glob(picklePath))

#colorBarの表示
plt.imshow(colorBar())
plt.savefig(dirPath+os.sep+"colorBar.png")

for ite,file in zip(range(len(files)),files):

    # pklファイルの読み込み
    # plot simulation data
    dataDict = pickle.load(open(picklePath,"rb"))
    XPO,YPO = dataDict["XPO"],dataDict["YPO"]
    Ts = np.array(dataDict["Ts"]) #shape=(23, 4374201)
    maxx = max(XPO)
    maxy = max(YPO)

    # マップ画像の初期化
    map = np.zeros([maxy,maxx])
    existImg = np.zeros([maxy,maxx]) # データの存在を０・１で示す
    sea = np.zeros([maxy,maxx])

    for i in range(layer-1):
        print("layer =", i)
        for j in range(Ts[0].shape[0]): #Ts[0].shape[0]=4374201

            x = XPO[j] - 1
            y = maxy - YPO[j]
            existImg[y,x] = 1
            sea[y,x] = 1
            map[y,x] += Ts[i][j]

        # img
        if ite==0:
            fig = plt.imshow(sea, cmap="gray", interpolation="None", vmin=0, vmax=1)
            plt.savefig(dirPath+os.sep+"sea.png",bbox_inches='tight')
            plt.clf()

        # img
        #pdb.set_trace()
        map_img = copy.deepcopy(map)
        map_img[sea == 0] = -1
        map_img[map_img<=0] -1
        fig = plt.imshow(bw2color(map_img), interpolation="None", vmin=0, vmax=255)
        plt.savefig(dirPath+os.sep+f"layer{layer}_upperDepth.png",bbox_inches='tight')
        plt.savefig(dirPath+os.sep+f"layer{layer}_upperDepth.pdf",bbox_inches='tight')
        plt.clf()

    # map
    map[sea == 0] = -1
    #cmap = plt.get_cmap("Reds")
    #cmap.set_under('grey')
    fig = plt.imshow(bw2color(map), interpolation="None", vmin=0, vmax=255)
    plt.savefig(dirPath+os.sep+f"layer{layer}_upperDepth_map.png",bbox_inches='tight')
    plt.clf()
