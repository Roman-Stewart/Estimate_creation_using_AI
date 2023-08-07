#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install tensorflow


# In[ ]:


pip install opencv-python


# In[ ]:


pip install scikit-learn


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # 최종 완성본

# In[1]:


import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split


import json

# JSON 파일 경로
json_file_path = 'C:/upload/IMG_OCR_53_4PO_09451.json'

# JSON 파일을 읽어와서 지정된 범위의 데이터를 추출
with open(json_file_path, 'r', encoding='utf-8') as json_file:
    json_lines = json_file.readlines()[29:-1]  # 30번째 줄부터 마지막 줄의 이전 줄까지

# 개행 문자를 제거하고 "bbox" 부분 제거하여 JSON 데이터로 파싱
json_data = ''.join(line.strip().replace('"bbox": ', '') for line in json_lines)
data = json.loads(json_data)


dict_list = data


# 이미지 로딩 및 전처리 함수 정의
def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 128))
    return image

# 이미지 데이터 및 레이블 준비
image = cv2.imread("C:/upload/image1.png")

# 바운딩 박스에 맞게 이미지 자르기
X = []
y = []
for dict in dict_list:
    x1 = min(dict["x"])
    x2 = max(dict["x"])
    y1 = min(dict["y"])
    y2 = max(dict["y"])
    box_image = image[y1:y2, x1:x2]
    box_image = cv2.cvtColor(box_image, cv2.COLOR_BGR2GRAY)
    box_image = cv2.resize(box_image, (128, 128))
    X.append(box_image)
    y.append(dict["data"])

# 그림 데이터 X와 라벨 데이터 y 순서에 맞게 배열에 넣기
X = np.array(X)
y = np.array(y)

X = X.reshape(-1, 128, 128, 1)


# 라벨 데이터 인코딩
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_onehot = to_categorical(y_encoded, num_classes=len(label_encoder.classes_))



# 데이터를 훈련 및 검증 세트로 분할
X_train, X_val, y_train, y_val = train_test_split(X, y_onehot, test_size=0.2, random_state=42)



# 모델 생성
model = keras.Sequential([
    layers.Input(shape=(128, 128, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])



# Early Stopping 콜백 정의 (학습 조기 종료)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 모델 훈련
model.fit(X_train, y_train, epochs=5000, validation_data=(X_val, y_val), callbacks=[early_stopping])


# 새로운 이미지에 대한 예측 함수 정의
def predict_new_image(image_path, model, label_encoder):
    new_image = load_and_preprocess_image(image_path)
    prediction = model.predict(np.array([new_image]))
    predicted_class = np.argmax(prediction)
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    return predicted_label

# 새 이미지 예측
new_image_path = 'C:/upload/image3.png'
predicted_label = predict_new_image(new_image_path, model, label_encoder)
print("Predicted Label:", predicted_label)


# In[ ]:


model.summary()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # XXXXXXXXXXXXXXXXXXXXXX 이전 버전XXXXXXXXXXXXXXXXXXXXX

# In[ ]:


import cv2 

image = cv2.imread("C:/upload/image1.png")


dict_list = [{'data': '09451',
  'id': 1,
  'x': [1848, 1848, 2019, 2019],
  'y': [199, 275, 199, 275]},
 {'data': '먹방', 'id': 2, 'x': [812, 812, 938, 938], 'y': [653, 726, 653, 726]},
 {'data': '공맹수',
  'id': 3,
  'x': [717, 717, 897, 897],
  'y': [832, 932, 832, 932]},
 {'data': '060',
  'id': 4,
  'x': [1729, 1729, 1821, 1821],
  'y': [769, 838, 769, 838]},
 {'data': '8978',
  'id': 5,
  'x': [1847, 1847, 1977, 1977],
  'y': [776, 837, 776, 837]},
 {'data': '1426',
  'id': 6,
  'x': [2008, 2008, 2131, 2131],
  'y': [770, 836, 770, 836]},
 {'data': '050',
  'id': 7,
  'x': [1730, 1730, 1816, 1816],
  'y': [864, 928, 864, 928]},
 {'data': '7156',
  'id': 8,
  'x': [1838, 1838, 1948, 1948],
  'y': [868, 925, 868, 925]},
 {'data': '3106',
  'id': 9,
  'x': [1967, 1967, 2083, 2083],
  'y': [870, 931, 870, 931]},
 {'data': '워라밸',
  'id': 10,
  'x': [749, 749, 919, 919],
  'y': [1163, 1251, 1163, 1251]},
 {'data': '030',
  'id': 11,
  'x': [1199, 1199, 1275, 1275],
  'y': [1211, 1271, 1211, 1271]},
 {'data': '9300',
  'id': 12,
  'x': [1296, 1296, 1391, 1391],
  'y': [1210, 1271, 1210, 1271]},
 {'data': '6752',
  'id': 13,
  'x': [1418, 1418, 1520, 1520],
  'y': [1211, 1274, 1211, 1274]},
 {'data': '090',
  'id': 14,
  'x': [1546, 1546, 1625, 1625],
  'y': [1210, 1280, 1210, 1280]},
 {'data': '7345',
  'id': 15,
  'x': [1649, 1649, 1763, 1763],
  'y': [1216, 1283, 1216, 1283]},
 {'data': '0115',
  'id': 16,
  'x': [1779, 1779, 1870, 1870],
  'y': [1217, 1274, 1217, 1274]},
 {'data': '유앙페',
  'id': 17,
  'x': [417, 417, 596, 596],
  'y': [1384, 1476, 1384, 1476]},
 {'data': '291962',
  'id': 18,
  'x': [1002, 1002, 1172, 1172],
  'y': [1374, 1449, 1374, 1449]},
 {'data': '326056',
  'id': 19,
  'x': [1279, 1279, 1455, 1455],
  'y': [1371, 1453, 1371, 1453]},
 {'data': '제주특별자치도',
  'id': 20,
  'x': [737, 737, 1044, 1044],
  'y': [1531, 1614, 1531, 1614]},
 {'data': '단양군',
  'id': 21,
  'x': [1057, 1057, 1203, 1203],
  'y': [1534, 1620, 1534, 1620]},
 {'data': '칠괴동',
  'id': 22,
  'x': [1222, 1222, 1361, 1361],
  'y': [1534, 1616, 1534, 1616]},
 {'data': '종로3가',
  'id': 23,
  'x': [1379, 1379, 1540, 1540],
  'y': [1534, 1619, 1534, 1619]},
 {'data': '인천광역시',
  'id': 24,
  'x': [746, 746, 966, 966],
  'y': [1633, 1716, 1633, 1716]},
 {'data': '수원시',
  'id': 25,
  'x': [986, 986, 1129, 1129],
  'y': [1642, 1715, 1642, 1715]},
 {'data': '사천시',
  'id': 26,
  'x': [1158, 1158, 1293, 1293],
  'y': [1641, 1714, 1641, 1714]},
 {'data': '함양군',
  'id': 27,
  'x': [1317, 1317, 1474, 1474],
  'y': [1639, 1718, 1639, 1718]},
 {'data': '99',
  'id': 28,
  'x': [1939, 1939, 2007, 2007],
  'y': [1545, 1615, 1545, 1615]},
 {'data': '383',
  'id': 29,
  'x': [2084, 2084, 2165, 2165],
  'y': [1550, 1619, 1550, 1619]},
 {'data': '671',
  'id': 30,
  'x': [344, 344, 444, 444],
  'y': [2485, 2549, 2485, 2549]},
 {'data': '714',
  'id': 31,
  'x': [527, 527, 624, 624],
  'y': [2486, 2553, 2486, 2553]},
 {'data': '법블레스유',
  'id': 32,
  'x': [961, 961, 1151, 1151],
  'y': [2486, 2552, 2486, 2552]},
 {'data': '93',
  'id': 33,
  'x': [1233, 1233, 1303, 1303],
  'y': [2489, 2554, 2489, 2554]},
 {'data': '3799',
  'id': 34,
  'x': [1332, 1332, 1464, 1464],
  'y': [2486, 2558, 2486, 2558]},
 {'data': '635377',
  'id': 35,
  'x': [1613, 1613, 1797, 1797],
  'y': [2483, 2558, 2483, 2558]},
 {'data': '937487',
  'id': 36,
  'x': [1841, 1841, 2047, 2047],
  'y': [2491, 2563, 2491, 2563]},
 {'data': '2041',
  'id': 37,
  'x': [337, 337, 444, 444],
  'y': [2645, 2704, 2645, 2704]},
 {'data': '31',
  'id': 38,
  'x': [495, 495, 565, 565],
  'y': [2650, 2706, 2650, 2706]},
 {'data': '93',
  'id': 39,
  'x': [631, 631, 707, 707],
  'y': [2645, 2704, 2645, 2704]},
 {'data': '4171',
  'id': 40,
  'x': [1284, 1284, 1401, 1401],
  'y': [2649, 2709, 2649, 2709]},
 {'data': '5317',
  'id': 41,
  'x': [1808, 1808, 1937, 1937],
  'y': [2650, 2714, 2650, 2714]},
 {'data': '9769',
  'id': 42,
  'x': [669, 669, 804, 804],
  'y': [2965, 3024, 2965, 3024]},
 {'data': '2032',
  'id': 43,
  'x': [1175, 1175, 1292, 1292],
  'y': [2965, 3028, 2965, 3028]},
 {'data': '53',
  'id': 44,
  'x': [1317, 1317, 1389, 1389],
  'y': [2974, 3028, 2974, 3028]},
 {'data': '46',
  'id': 45,
  'x': [1416, 1416, 1483, 1483],
  'y': [2971, 3030, 2971, 3030]},
 {'data': '유도순',
  'id': 46,
  'x': [455, 455, 594, 594],
  'y': [3103, 3188, 3103, 3188]},
 {'data': '474733',
  'id': 47,
  'x': [616, 616, 773, 773],
  'y': [3109, 3179, 3109, 3179]},
 {'data': '687987',
  'id': 48,
  'x': [774, 774, 909, 909],
  'y': [3115, 3188, 3115, 3188]},
 {'data': '누나',
  'id': 49,
  'x': [1114, 1114, 1222, 1222],
  'y': [3117, 3196, 3117, 3196]},
 {'data': '여',
  'id': 50,
  'x': [1385, 1385, 1482, 1482],
  'y': [3113, 3188, 3113, 3188]},
 {'data': '탁정라',
  'id': 51,
  'x': [453, 453, 599, 599],
  'y': [3208, 3285, 3208, 3285]},
 {'data': '335294',
  'id': 52,
  'x': [613, 613, 751, 751],
  'y': [3219, 3289, 3219, 3289]},
 {'data': '682536',
  'id': 53,
  'x': [774, 774, 908, 908],
  'y': [3221, 3278, 3221, 3278]},
 {'data': '손',
  'id': 54,
  'x': [1126, 1126, 1203, 1203],
  'y': [3219, 3290, 3219, 3290]},
 {'data': '부',
  'id': 55,
  'x': [1396, 1396, 1466, 1466],
  'y': [3211, 3293, 3211, 3293]},
 {'data': '주', 'id': 56, 'x': [755, 755, 790, 790], 'y': [656, 710, 656, 710]},
 {'data': '주',
  'id': 57,
  'x': [689, 689, 727, 727],
  'y': [1164, 1227, 1164, 1227]},
 {'data': '주',
  'id': 58,
  'x': [913, 913, 944, 944],
  'y': [2498, 2544, 2498, 2544]}]


from tensorflow import keras 


X = [] 
y = []
for dict in dict_list:
    x1 = min(dict["x"]) 
    x2 = max(dict["x"]) 
    y1 = min(dict["y"]) 
    y2 = max(dict["y"]) 
    box_image = image[y1:y2,x1:x2]
    box_image = cv2.cvtColor(box_image,cv2.COLOR_BGR2GRAY) 
    box_image = cv2.resize(box_image,(64,64)) 
    X.append(box_image)
    y.append(dict["data"])


import numpy as np 
X = np.array(X) 
y = np.array(y)
             
             
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
X = X.reshape(-1,64,64,1)
             
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 모델 생성
model = keras.Sequential([
    layers.Input(shape=(64, 64, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(labels), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 모델 훈련
model.fit(X, y_encoded, batch_size = 32, epochs=50)


def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (64, 64))
    return image


# 새로운 이미지에 대한 예측
new_image = load_and_preprocess_image('C:/upload/image3.png')
prediction = model.predict(np.array([new_image]))
predicted_label = labels[np.argmax(prediction)]

print("Predicted Label:", predicted_label)


# In[ ]:


model.summary()


# In[ ]:


model = keras.models.load_model(r"C:/upload/handwriting_model.h5")


# In[ ]:


model.save(r"C:/upload/handwriting_model.h5")


# In[ ]:





# In[ ]:


import json

# JSON 파일 경로
json_file_path = 'C:/upload/IMG_OCR_53_4PO_09451.json'

# JSON 파일을 읽어와서 지정된 범위의 데이터를 추출
with open(json_file_path, 'r', encoding='utf-8') as json_file:
    json_lines = json_file.readlines()[29:-1]  # 30번째 줄부터 마지막 줄의 이전 줄까지

# 개행 문자를 제거하고 "bbox" 부분 제거하여 JSON 데이터로 파싱
json_data = ''.join(line.strip().replace('"bbox": ', '') for line in json_lines)
data = json.loads(json_data)
data


# In[ ]:


from tensorflow.keras.preprocessing.text import Tokenizer

# 원래 한글 문장
original_text = "안녕하세요, 한글 tokenizer 인코딩 예제입니다."

# tokenizer 객체 생성
tokenizer = Tokenizer(char_level=True)

# tokenizer를 학습하여 문장을 숫자로 인코딩
tokenizer.fit_on_texts([original_text])

# 인코딩된 데이터 생성
encoded_data = tokenizer.texts_to_sequences([original_text])[0]

# 인코딩된 데이터 출력
print(encoded_data)


# In[ ]:


# 인덱스를 문자로 변환하여 디코딩된 문장 생성
decoded_text = tokenizer.sequences_to_texts([encoded_data])[0]

# 디코딩된 문장 출력
print(decoded_text)


# In[ ]:





# In[ ]:


import tensorflow as tf
import json
from tensorflow.keras.preprocessing.text import Tokenizer

# JSON 파일 경로
json_file_path = 'C:/upload/IMG_OCR_53_4PO_09451.json'

# JSON 파일을 읽어와서 지정된 범위의 데이터를 추출
with open(json_file_path, 'r', encoding='utf-8') as json_file:
    json_lines = json_file.readlines()[29:-1]  # 30번째 줄부터 마지막 줄의 이전 줄까지

# 개행 문자를 제거하고 "bbox" 부분 제거하여 JSON 데이터로 파싱
source_json_data = ''.join(line.strip().replace('"bbox": ', '') for line in json_lines)
json_data = json.loads(source_json_data)

# Tokenizer 생성 및 데이터 학습
tokenizer = Tokenizer(char_level=True)
data = [item["data"] for item in json_data]
tokenizer.fit_on_texts(data)

# 데이터 인코딩
new_data = []
encoded_data_list = []
for item in json_data:
    encoded_data = tokenizer.texts_to_sequences([item["data"]])[0]
    new_item = {
        "data": encoded_data,
        "id": item["id"],
        "x": item["x"],
        "y": item["y"]
    }
    new_data.append(new_item)

# 결과 출력 (예시로 처음 5개 데이터만 출력)
for idx, item in enumerate(new_data[:5]):
    print(f"Data {idx + 1}:")
    print("  'data':", item["data"])
    print("  'id':", item["id"])
    print("  'x':", item["x"])
    print("  'y':", item["y"])


# In[ ]:


# 디코딩 함수 정의
def decode_sequence(encoded_sequence):
    return tokenizer.sequences_to_texts([encoded_sequence])[0]

# 결과 출력 및 디코딩 (예시로 처음 5개 데이터만 출력)
for idx, item in enumerate(new_data[:5]):
    print(f"Data {idx + 1}:")
    print("  'data':", decode_sequence(item["data"]))
    print("  'id':", item["id"])
    print("  'x':", item["x"])
    print("  'y':", item["y"])


# In[ ]:


data


# In[ ]:


new_data


# In[ ]:




