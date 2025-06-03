import numpy as np
import cv2
from PIL import Image

def getContours(base_img):

    # Calcular contornos de la imagen
    contours, _ = cv2.findContours(image=base_img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    dim = np.shape(base_img)

    contours_img = cv2.drawContours(np.zeros((dim[0],dim[1],3)), contours, -1, (0,255,0), 1)
    contours_img = np.uint8(contours_img)

    return contours, contours_img

def filterContours(base_img, contours, mult=2.5, mode='DOWN'):
    # Filtrado de contornos por longitud
    contours_lenght = [len(c) for c in contours]

    cl_mean = np.mean(contours_lenght)
    cl_std = np.std(contours_lenght)

    if mode == 'DOWN':
        contours_filtered = [c for c in contours if len(c) < cl_mean + mult*cl_std]
    elif mode == 'UP':
        contours_filtered = [c for c in contours if len(c) > cl_mean + mult*cl_std]

    dim = np.shape(base_img)
    contours_filtered_img = cv2.drawContours(np.zeros((dim[0],dim[1],3)), contours_filtered, -1, (0,255,0), 1)
    contours_filtered_img = np.uint8(contours_filtered_img)

    return contours_filtered, contours_filtered_img

def getIndividualMasks(base_mask,contours):
    # Lista para guardar las máscaras individuales
    individual_masks = []
    
    # Crear una máscara por cada contorno
    for i, contour in enumerate(contours):
        # Máscara en negro (0)
        mask = np.zeros_like(base_mask, dtype=np.uint8)
        
        # Dibujar el contorno RELLENO (cv2.FILLED o thickness=-1)
        cv2.drawContours(
            image=mask,
            contours=[contour],
            contourIdx=-1,  # -1 para dibujar todos los contornos en la lista
            color=255,     # Blanco (255)
            thickness=cv2.FILLED  # Rellenar el contorno
        )
        
        individual_masks.append(mask)
    
    return individual_masks

def getBoundingBox(mask):

    # Conversión del la máscara a clase Image
    mask_class = Image.fromarray(mask)

    # Obtención del bounding box
    bbox = mask_class.getbbox()

    # Conversión de bbox a formato utilizable
    x1,y1,x2,y2 = bbox

    bbox_coors = [(x1,y1),(x2,y2)]

    return bbox_coors