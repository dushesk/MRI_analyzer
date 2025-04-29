import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import os
from datetime import datetime
import cv2
import base64
from PIL import Image

class GradCAM:
    """Работа с Grad-CAM heatmap"""
    
    @staticmethod
    def generate_heatmap(model, img_array, layer_name='conv2d_5'):
        """Генерация heatmap"""
        # Создаем подмодель: вход → выбранный слой + выход модели
        grad_model = Model(
            inputs=model.inputs,
            outputs=[model.get_layer(layer_name).output, model.outputs[0]]
        )

        img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)

        # Вычисляем градиенты
        with tf.GradientTape() as tape:
            tape.watch(img_array)
            conv_outputs, predictions = grad_model(img_array)            
            class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]
        
        grads = tape.gradient(loss, conv_outputs)

        # Усредняем градиенты по пространственным осям
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Создаем heatmap
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
        
        return heatmap.numpy()  

    @staticmethod
    def save_heatmap(heatmap, save_dir=os.path.join("static","gradcam")):
        """Сохранение heatmap"""
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"heatmap_{timestamp}.png")

        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Сохраняем как grayscale
        cv2.imwrite(save_path, heatmap)
        return save_path

    @staticmethod
    def prepare_heatmap_image(heatmap):
        """Подготовка heatmap"""
        heatmap = 1 - heatmap   # Реверсия

        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)        
        return Image.fromarray(heatmap)
