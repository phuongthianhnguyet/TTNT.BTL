## TTNT.BTL
### Link Youtube: ```

### Link dữ liệu : ``` https://download.microsoft.com/download/3/e/1/3e1c3f21-ecdb-4869-8368-6deba77b919f/kagglecatsanddogs_5340.zip ```

### Hướng dẫn cài đặt và chạy chương trình.
Đăng nhập vào google colab và tạo sổ tay mới
Bật GPU lên để chuẩn bị chạy chương trình

Bước 1: Tải dữ liệu chó và mèo từ link lên google colab và giải nén dữ liệu

```
!wget https://download.microsoft.com/download/3/e/1/3e1c3f21-ecdb-4869-8368-6deba77b919f/kagglecatsanddogs_5340.zip -O cats_dogs.zip
!unzip -q cats_dogs.zip -d dataset
```


Bước 2: Tiền xử lý dữ liệu (Data Preprocessing)(làm sạch dữ liệu)

```
import os
import shutil
from PIL import Image

base_dir = 'dataset/PetImages'
cat_dir = os.path.join(base_dir, 'Cat')
dog_dir = os.path.join(base_dir, 'Dog')

def cleanup_corrupt_images(directory):
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            # Remove Thumbs.db files which are not images
            if filename.lower() == 'thumbs.db':
                print(f"Removing Thumbs.db: {filepath}")
                os.remove(filepath)
                continue
            # Remove zero-byte files
            if os.path.getsize(filepath) == 0:
                print(f"Removing zero-byte file: {filepath}")
                os.remove(filepath)
                continue
            # Try to open image with PIL to detect corruption
            try:
                img = Image.open(filepath)
                img.verify() # Verify that it is, in fact, an image
                img.close()
            except Exception as e:
                print(f"Removing corrupt image: {filepath} ({e})")
                os.remove(filepath)

print("Cleaning Cat directory...")
cleanup_corrupt_images(cat_dir)
print("Cleaning Dog directory...")
cleanup_corrupt_images(dog_dir)

# Đếm số ảnh
print(len(os.listdir(cat_dir)), "ảnh mèo")
print(len(os.listdir(dog_dir)), "ảnh chó")
```

Bước 3: TẠO DATASET & CHIA DỮ LIỆU
```
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Tạo dataset từ thư mục (tự động chia train/val)
train_ds = image_dataset_from_directory(
    base_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(150, 150),  # resize về 150x150
    batch_size=32
)

val_ds = image_dataset_from_directory(
    base_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(150, 150),
    batch_size=32
)

class_names = train_ds.class_names  # ['Cat', 'Dog']
```

Bước 4: Xây dựng mô hình (Model Building) + Cấu hình huấn luyện
```
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(150, 150, 3)),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # binary → sigmoid
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
```

Bước 5: Vẫn thuộc bước tiền xử lý dữ liệu nhưng là nâng cao hơn.
```
import os
from PIL import Image
import tensorflow as tf

base_dir = 'dataset/PetImages'   # <-- giữ nguyên như code cũ của bạn

def remove_corrupted_images(base_dir):
    removed = 0
    for folder in ['Cat', 'Dog']:
        folder_path = os.path.join(base_dir, folder)
        for fname in list(os.listdir(folder_path)):   # list() để tránh lỗi khi xóa
            fpath = os.path.join(folder_path, fname)
            try:
                # Kiểm tra bằng PIL
                with Image.open(fpath) as img:
                    img.verify()
                # Kiểm tra bằng TensorFlow
                img_data = tf.io.read_file(fpath)
                tf.io.decode_jpeg(img_data, channels=3)
            except Exception:
                print(f"🗑️  Xóa ảnh hỏng: {fname}")
                os.remove(fpath)
                removed += 1
    print(f"✅ Đã dọn dẹp xong! Xóa tổng cộng {removed} ảnh hỏng.")
    return removed

# CHẠY ĐOẠN NÀY
remove_corrupted_images(base_dir)

# Kiểm tra số lượng còn lại
print("Mèo còn lại:", len(os.listdir(os.path.join(base_dir, 'Cat'))))
print("Chó còn lại:", len(os.listdir(os.path.join(base_dir, 'Dog'))))
```

Bước 6: Bước huấn luyện mô hình (Training),Callback (kỹ thuật hỗ trợ training)
```
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
```

Bước 7: Huấn luyện mô hình.
```
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10  # tăng lên 15-20 nếu muốn accuracy cao hơn
)
```

Bước 8: TRANSFER LEARNING (MOBILENETV2)
```
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Re-create datasets to ensure they are defined and can be resized
# base_dir is defined in cell B-B3puynh-66, so we'll assume it's available.
# If not, it would need to be re-defined here as well: base_dir = 'dataset/PetImages'

train_ds = image_dataset_from_directory(
    base_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(150, 150),  # Keep original size for initial loading
    batch_size=32
)

val_ds = image_dataset_from_directory(
    base_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(150, 150),
    batch_size=32
)

# Resize datasets to 224x224 to match MobileNetV2's preferred input
train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, [224, 224]), y))
val_ds   = val_ds.map(lambda x, y: (tf.image.resize(x, [224, 224]), y))


base_model = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), # Changed to 224x224
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the MobileNetV2 model and assign its history
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10  # You can adjust the number of epochs
)
```

Bước 9: Biểu đồ kết quả.
```
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(acc, label='train')
plt.plot(val_acc, label='val')
plt.legend()
plt.show()
```

Bước 10: Lưu và tải file hoàn chỉnh về thư mục.
```
model.save("cats_dogs_model.h5")
from google.colab import files
files.download("cats_dogs_model.h5")
```
Chúng ta có thể test trực tiếp trên colab nhưng em chọn tạo giao diện riêng. Code giao diện python:
```
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("cats_dogs_model.h5")

IMG_SIZE = 224  # nếu bạn dùng MobileNetV2

def predict_image(file_path):
    img = Image.open(file_path).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        result = f"🐶 Chó ({prediction:.2f})"
    else:
        result = f"🐱 Mèo ({1 - prediction:.2f})"
    
    return result, img


def upload_image():
    file_path = filedialog.askopenfilename()
    
    if not file_path:
        return

    result, img = predict_image(file_path)

    # Resize ảnh để hiển thị
    img_display = img.resize((150, 150))
    img_tk = ImageTk.PhotoImage(img_display)

    panel.config(image=img_tk)
    panel.image = img_tk

    label_result.config(text=result)


# GUI
root = tk.Tk()
root.title("🐱🐶 Chó hay Mèo")
root.geometry("400x500")

btn = tk.Button(root, text="Chọn ảnh", command=upload_image)
btn.pack(pady=10)

panel = tk.Label(root)
panel.pack()

label_result = tk.Label(root, text="", font=("Arial", 16))
label_result.pack(pady=20)

root.mainloop()
```
### THE END 


