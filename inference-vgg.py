import cv2
from VGG16 import my_vgg16
from torchvision import transforms
import torch
from PIL import Image
import numpy as np
import os
import pandas as pd
from sklearn.cluster import KMeans

'''
BATCH_SIZE = 2
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
model_path = 'save_model/vgg_model1.pth'

data_transform = transforms.Compose([
   transforms.Resize((244, 244)),
   transforms.ToTensor(),
   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

#模型加载
preNet = my_vgg16(numClass=2)
preNet.load_state_dict(torch.load(model_path, map_location=device))
preNet.to(device)

#模型结果存储为csv
IMG_PATH = 'images/'
df_id = pd.read_csv('submission.csv')
id = df_id.iloc[:,0]
id = np.array(id)
k = []
for index,pic in enumerate(id):  
   tmpImg = cv2.imread(IMG_PATH+pic)
   tmpImg = cv2.resize(tmpImg,(640,480))
   tmp = Image.fromarray(cv2.cvtColor(tmpImg,cv2.COLOR_BGR2RGB))
   img = data_transform(tmp).unsqueeze(0) 
   img = img.to(device)
   testlable = preNet(img)
   check = testlable.data.cpu().numpy()
   k.append(check[0][0])#只用第一列数据
cls = KMeans(n_clusters = 2,random_state=20221112)#设置种子为20221112，防止结果不一致
k = np.array(k)
k = k.reshape(-1,1)
X = k[:]
data = cls.fit(k)
pred = data.labels_
for i in range(len(X)):
   if pred[i] == 1: 
      df_id.iloc[i,1] = 'rainy'
   if pred[i] == 0:
      df_id.iloc[i,1] = 'normal'
#结果保存
try:
   df_id.to_csv('submission_.csv',index = False)#去掉索引
   print('已保存为submission_.csv')
except:
   print('保存时出现了未知错误，请检查当前目录是否有重名文件')
   pass
   
      