# DL_Vehicle-Type-Identification_CNN


## STEP1. 데이터 전처리

* 정규화 및 One-Hot 벡터 전환

X_train = X_train.reshape(-1, 128, 128, 1).astype('float32') / 255
X_test = X_test.reshape(-1, 128, 128, 1).astype('float32') / 255
-> 0~1 사이의 실수값으로 변환

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
-> 11개의 범주형 자료를 one-hot 벡터로 변환

## 데이터 분석

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential

activation = 'relu'
initializer = 'he_normal'
dropout = 0.3
optimizer = 'adam'

model = Sequential() # Sequential API 사용
model.add(Conv2D(32, (3,3), activation = activation, kernel_initializer=initializer, input_shape = [128,128,1]) # 모수 : 32 x (3x3 +1) = 320; 사이즈는 126x126

model.add(MaxPooling2D(2,2)) # 2x2 MaxPooling으로 이미지 사이즈는 반으로 줄음 63x63 노드는 32개로 유지

model.add(Conv2D(64, (3,3), activation = activation, kernel_initializer = initializer, padding='same')) # 이미지 사이즈를 반으로 줄였기 때문에 노드를 2배 증가한 64개로 설정; padding='same' 옵션을 사용하여 원래의 크기 유지; size는 61x61; 모수 64x(32x9 +1) = 18496

model.add(Maxpooling2D(2,2))

model.add(Flatten()) # Flatten() 함수를 사용하여 1D 텐서로 재배치; 31x31x64 = 61504
model.add(Dropout(dropout)) #정규화
model.add(Dense(11, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', tf.keras.metrics.Recall()])
history2 = model.fit(X_train, y_train, batch_size=64, epochs=5, validation_split=0.1)
score2 = model.evaluate(X_test, y_test) # test data의 결과

## 분석 결과

- Layers : 32-64
- Activation : Relu / LeakyRelu / SELU
- Optimizer : Adam / RMSprop
- Dropout : 0.3 / 0.5
-> 다음과 같은 Hyperparameter를 설정하여 Loss / Accuracy / Recall 값을 비교하여 가장 좋은 결과가 나온 Hyperparameter 선정
