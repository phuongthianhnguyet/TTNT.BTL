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