import numpy as np
import cv2
import threading

from transformers import ViTImageProcessor, ViTModel
import torch
from skimage.feature import graycomatrix, graycoprops, hog

import tensorflow as tf

"""EMBEDDING MODEL"""
pretrained_vit_feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
pretrained_vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

"""CNN MODEL"""
# i have a model saved in the parent directory
cnn_model = tf.keras.models.load_model('fine_tuned_model.h5')


def create_color_histogram(img, bins=8):
    # PIL Image object to OpenCV image
    img = np.array(img)

    hist_r, _ = np.histogram(img[:, :, 0], bins=bins, range=(0, 255))
    hist_g, _ = np.histogram(img[:, :, 1], bins=bins, range=(0, 255))
    hist_b, _ = np.histogram(img[:, :, 2], bins=bins, range=(0, 255))
    
    new_vector = []
    for freqR in hist_r:
        for freqG in hist_g:
            for freqB in hist_b:
                new_vector.append(freqR + freqG + freqB)

    return np.array([new_vector], dtype=np.float32)

def get_glcm_features(img, distances = [1, 3, 5, 7], angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calcular la GLCM
    glcm = graycomatrix(img, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
   
    # Extraer características: contraste, correlación, energía y homogeneidad
    contrast = graycoprops(glcm, 'contrast').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
    asm = graycoprops(glcm, 'ASM').flatten()
    dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
   
    # Combinar todas las características en un solo vector
    features = np.hstack([contrast, correlation, energy, homogeneity, asm, dissimilarity])
    return np.array([features], dtype=np.float32)

def extract_hog_features(img):
    img = np.array(img)

    # Función para preprocesar la imagen
    def preprocess_image(img):
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (224, 224))
        return image
    
    prep_image = preprocess_image(img)
    features, _ = hog(prep_image, orientations=5, pixels_per_cell=(24, 24), cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
    return np.array([features], dtype=np.float32)

def extract_cnn_features(img):
    image = np.array(img)
    image = image / 255.0
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    embedding = cnn_model.predict(image)
    return np.array([embedding.flatten()], dtype=np.float32)

def create_vit_embedding(img):
    img = np.array(img)
    
    # Función para cargar y preparar la imagen
    def preprocess_image(image):
        inputs = pretrained_vit_feature_extractor(images=image, return_tensors="pt")
        return inputs

    # Genera el embedding de la imagen
    def get_image_embedding(image_path):
        inputs = preprocess_image(image_path)
        with torch.no_grad():
            outputs = pretrained_vit_model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :]  # Usamos el embedding de la clase [CLS]
        return embedding
    
    return np.array([get_image_embedding(img)], dtype=np.float32).reshape(1, 768)