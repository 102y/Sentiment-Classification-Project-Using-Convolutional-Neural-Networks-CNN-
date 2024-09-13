import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# تحميل بيانات FER2013 من كاغل
data = pd.read_csv('fer2013.csv')

# معالجة البيانات
def load_data(data):
    images = []
    labels = []
    
    for index, row in data.iterrows():
        pixels = np.fromstring(row['pixels'], dtype=int, sep=' ')
        image = pixels.reshape(48, 48)
        image = np.expand_dims(image, axis=-1)  # إضافة بعد ثالث للصورة
        images.append(image)
        labels.append(row['emotion'])
    
    return np.array(images), np.array(labels)

X, y = load_data(data)
X = X / 255.0  # تطبيع الصور

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# بناء نموذج الـ CNN
model = Sequential()

# الطبقة الالتفافية الأولى
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D((2, 2)))

# الطبقة الالتفافية الثانية
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# الطبقة الالتفافية الثالثة
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# تحويل المصفوفة إلى متجه
model.add(Flatten())

# الطبقة الكاملة الربط
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# الطبقة النهائية
model.add(Dense(7, activation='softmax'))  # لدينا 7 فئات للعواطف

# تجميع النموذج
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# تدريب النموذج
model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test))

# حفظ النموذج بعد التدريب
model.save('emotion_detection_model.h5')

# دالة لتوقع العاطفة من الصورة وعرضها مع النص
def predict_emotion(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (48, 48))
    img_expanded = np.expand_dims(img_resized, axis=-1)  # إضافة بعد ثالث للصورة
    img_batch = np.expand_dims(img_expanded, axis=0)   # إضافة بعد للدفعات (batch)
    img_normalized = img_batch / 255.0  # تطبيع الصورة
    
    predictions = model.predict(img_normalized)
    emotion = np.argmax(predictions)
    
    emotion_dict = {0: 'sinirli', 1: 'tiksinti', 2: 'korkmuş', 3: 'mutlu', 4: 'üzgün', 5: 'Şaşırmış', 6: 'doğal'}
    emotion_text = emotion_dict[emotion]
    
    # عرض الصورة مع النص
    plt.imshow(img, cmap='gray')
    plt.title(f"Beklenen duygu : {emotion_text}")
    plt.axis('off')  # إخفاء المحاور
    plt.show()

# استخدام البرنامج للتنبؤ بالعواطف من صورة
image_path = 'C:\\Users\\NITRO\\Desktop\\AI Proj\\Recognizing emotions in images\\images.jpg'
predict_emotion(image_path)
