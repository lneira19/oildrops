# Módulos propios
import imgproc.basics as ipbasics

# Módulos externos
import numpy as np
import cv2
import pandas as pd
from tensorflow.keras.models import load_model

class Bbox:
    def __init__(self,
                 hsv_lower_limit=np.array([138, 40, 110]),
                 hsv_upper_limit=np.array([178, 250, 255])):

        self.hsv_lower_limit =  hsv_lower_limit
        self.hsv_upper_limit = hsv_upper_limit
        self.pixel_area = 1/1936 #mm2
    
    def getBoundingBoxesForImg(self,bgr_img,sdk=-1,minarea=250):

        # Convertir la imagen BGR a HSV
        hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

        # Aplicar desenfoque gaussiano para reducir el ruido
        hsv_img_blur = cv2.GaussianBlur(hsv_img, (3, 3), 0)

        # rosa base = [169,35,90]
        lower_color = self.hsv_lower_limit 
        upper_color = self.hsv_upper_limit  

        # Crear una máscara para el color rosa
        mask = cv2.inRange(hsv_img_blur, lower_color, upper_color)

        if np.sum(mask) == 0:
            return [], []
        else:
            # Aplicar apertura morfológica para eliminar ruido
            opening_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, opening_kernel)

            # Obtener contornos de la máscara
            contours, contours_img = ipbasics.getContours(mask_open)

            # Filtrar contornos por longitud
            contours_filtered, contours_filtered_img = ipbasics.filterContours(mask_open, contours, mult=sdk, mode='UP')

            # Obtener máscaras individuales para cada contorno filtrado
            list_individual_masks = ipbasics.getIndividualMasks(mask_open,contours_filtered)

            # Obtener bounding boxes para cada máscara individual
            list_bboxes = [ipbasics.getBoundingBox(mask) for mask in list_individual_masks]

            # Filtrar bounding boxes y máscaras individuales por tamaño válido de bounding box
            list_individual_masks_filtered = [mask for mask, bbox in zip(list_individual_masks, list_bboxes) if (bbox[0][0] - bbox[1][0])*(bbox[0][1] - bbox[1][1]) > minarea]
            list_bboxes_filtered = [bbox for bbox in list_bboxes if (bbox[0][0] - bbox[1][0])*(bbox[0][1] - bbox[1][1]) > minarea]

            # Ordenar bbox y individualmask por x1 (primer elemento del primer punto) de la bbox
            list_bboxes_filtered_sorted = sorted(list_bboxes_filtered, key=lambda bbox: bbox[0][0])
            list_individual_masks_filtered_sorted = [mask for _, mask in sorted(zip(list_bboxes_filtered, list_individual_masks_filtered), key=lambda x: x[0][0][0])]

            return list_bboxes_filtered_sorted, list_individual_masks_filtered_sorted
    
    def getImgWithBboxes(self, path_img, sdk=-1,minarea=250):
        
        bgr_img = cv2.imread(path_img) # Imagen BGR
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB) # Convertir a RGB para matplotlib

        bboxes, masks = self.getBoundingBoxesForImg(bgr_img=bgr_img,sdk=sdk,minarea=minarea)

        if len(bboxes) == 0:
            return rgb_img, []
        else:
            rgb_img_copy = rgb_img.copy()
            for bbox in bboxes:
                cv2.rectangle(rgb_img_copy, bbox[0], bbox[1], (0, 255, 0), 2)
        
        return rgb_img_copy, bboxes
    
    def getDataFrameForImg(self, path_img, name_img, sdk=-1, minarea=250):
        
        bgr_img = cv2.imread(path_img) # Imagen BGR

        # Obtener bounding boxes y máscaras
        bboxes, masks = self.getBoundingBoxesForImg(bgr_img,sdk=sdk,minarea=minarea)

        # Crear un DataFrame para almacenar los resultados
        df = pd.DataFrame(columns=['file_name','bbox','x1', 'y1', 'x2', 'y2', 'width', 'height','bbox_area', 'mask_area'])

        # Si no se encuentran bounding boxes, retornar un DataFrame vacío
        if len(bboxes) == 0:
            df.loc[len(df)] = [name_img, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        else:
            # Iterar sobre las bounding boxes y máscaras para llenar el DataFrame
            for i, pack in enumerate(zip(bboxes, masks)):
        
                bbox, mask = pack

                x1, y1 = bbox[0]
                x2, y2 = bbox[1]
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                bbox_area = bbox_width * bbox_height
                mask_area = int(np.sum(mask) / 255)

                df.loc[len(df)] = [name_img, i, x1, y1, x2, y2, bbox_width, bbox_height, bbox_area, mask_area]

        return df
    
    def getImgWithBboxesAndDrops(self, path_img, sdk=-1,minarea=250,model_to_use="dropcounter_v02_model"):
        
        # Cargar el modelo previamente entrenado
        loaded_model = load_model('../models/'+model_to_use+'.keras')

        bgr_img = cv2.imread(path_img) # Imagen BGR
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB) # Convertir a RGB
        bboxes, masks = Bbox().getBoundingBoxesForImg(bgr_img, sdk=sdk, minarea=minarea)

        rgb_img_copy = rgb_img.copy()
        predictions = []
        areas = []

        if len(bboxes) == 0:
            rgb_img_copy = rgb_img
        else:
            for i, pack in enumerate(zip(bboxes, masks)):
                bbox, mask = pack
                
                # Área de la máscara
                area_mask = int(np.sum(mask) / 255) * self.pixel_area
                areas.append(area_mask)

                # Features
                x1, y1 = bbox[0]
                x2, y2 = bbox[1]
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                bbox_area = bbox_width * bbox_height
                mask_area = int(np.sum(mask) / 255)

                features = np.array([[x1/800, y1/600, x2/800, y2/600, bbox_width/800, bbox_height/600, bbox_area/(800*600), mask_area/(800*600)]])

                # Predicción
                prediction = loaded_model.predict(features, verbose=0)
                prediction = np.uint8(np.round(prediction[0][0]))
                predictions.append(prediction)

                cv2.rectangle(rgb_img_copy, bbox[0], bbox[1], (0, 255, 0), 2)

                # Colocación de texto en la imagen
                text = f"D:{prediction} A:{area_mask:.2f}"
                cv2.putText(rgb_img_copy, text, (bbox[0][0], bbox[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return rgb_img_copy, predictions, areas