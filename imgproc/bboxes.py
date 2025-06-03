# Módulos propios
import imgproc.basics as ipbasics

# Módulos externos
import numpy as np
import cv2

class Bbox:
    def __init__(self,
                 hsv_lower_limit=np.array([138, 40, 110]),
                 hsv_upper_limit=np.array([178, 250, 255])):

        self.hsv_lower_limit =  hsv_lower_limit
        self.hsv_upper_limit = hsv_upper_limit
    
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

        # Filtrar bounding boxes que no tengan un área mínima
        list_bboxes = [bbox for bbox in list_bboxes if (bbox[0][0] - bbox[1][0]) * (bbox[0][1] - bbox[1][1]) > minarea]

        return list_bboxes
    
    def getImgWithBboxes(self, path_img, sdk=-1,minarea=250):
        
        bgr_img = cv2.imread(path_img) # Imagen BGR
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB) # Convertir a RGB para matplotlib

        bboxes = self.getBoundingBoxesForImg(bgr_img=bgr_img,sdk=sdk,minarea=minarea)

        rgb_img_copy = rgb_img.copy()
        for bbox in bboxes:
            cv2.rectangle(rgb_img_copy, bbox[0], bbox[1], (0, 255, 0), 2)
        
        return rgb_img_copy, bboxes