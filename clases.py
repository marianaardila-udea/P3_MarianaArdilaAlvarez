
import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
import nibabel as nib
import pandas as pd
import os

class EstudioImaginologico:
    def __init__(self, study_date, study_time, modality, description, series_time, duration, image_3d, pixel_spacing, slice_thickness):
        self.study_date = study_date
        self.study_time = study_time
        self.modality = modality
        self.study_description = description
        self.series_time = series_time
        self.duration = duration
        self.image_3d = image_3d
        self.shape = image_3d.shape
        self.pixel_spacing = pixel_spacing
        self.slice_thickness = slice_thickness
    
    def mostrar_reconstruccion_3d(self):
        mid = [s // 2 for s in self.shape]
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        # Transversal
        axs[0].imshow(self.image_3d[mid[0], :, :], cmap="gray")
        axs[0].set_title("Corte Transversal")
        axs[0].axis("off")
        
        # Sagital
        axs[1].imshow(np.rot90(self.image_3d[:, mid[1], :]), cmap="gray")
        axs[1].set_title("Corte Sagital")
        axs[1].axis("off")
        
        # Coronal
        axs[2].imshow(np.rot90(self.image_3d[:, :, mid[2]]), cmap="gray")
        axs[2].set_title("Corte Coronal")
        axs[2].axis("off")
        
        plt.suptitle(f"{self.study_description} - {self.modality}")
        plt.show()
    def zoom(self):
        z = int(input(f"Ingrese índice de corte transversal (0-{self.shape[0]-1}): "))
        slice_2d = self.image_3d[z, :, :].astype(float)
        
        img_norm = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min()) * 255
        img_uint8 = img_norm.astype(np.uint8)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)

        x1, y1, x2, y2 = map(int, input("Ingrese x1 y1 x2 y2 (columna fila): ").split())
        
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        width_mm = abs(x2 - x1) * self.pixel_spacing[1]
        height_mm = abs(y2 - y1) * self.pixel_spacing[0]
        text = f"{width_mm:.2f}x{height_mm:.2f} mm"
        cv2.putText(img_bgr, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        crop = img_bgr[y1:y2, x1:x2]
        scale = float(input("Factor de redimensionamiento (>1 para zoom): "))
        resized = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)))

        fig, axs = plt.subplots(1, 2, figsize=(14, 7))
        axs[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Original")
        axs[1].imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        axs[1].set_title("Recortada y redimensionada")
        plt.show()

        nombre = input("Nombre para guardar redimensionada (.png): ")
        cv2.imwrite(nombre, resized)
    def segmentacion(self):
        z = int(input(f"Ingrese índice de corte (0-{self.shape[0]-1}): "))
        slice_2d = self.image_3d[z, :, :].astype(float)
        img_norm = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min()) * 255
        img_uint8 = img_norm.astype(np.uint8)

        print("1: Binario\n2: Binario invertido\n3: Truncado\n4: To zero\n5: To zero invertido")
        tipo = int(input("Tipo: "))
        tipos = {1: cv2.THRESH_BINARY, 2: cv2.THRESH_BINARY_INV, 3: cv2.THRESH_TRUNC,
                 4: cv2.THRESH_TOZERO, 5: cv2.THRESH_TOZERO_INV}
        
        thresh = float(input("Umbral (0 para Otsu): "))
        if thresh == 0:
            _, seg = cv2.threshold(img_uint8, 0, 255, tipos[tipo] + cv2.THRESH_OTSU)
        else:
            _, seg = cv2.threshold(img_uint8, thresh, 255, tipos[tipo])

        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(img_uint8, cmap="gray"); axs[0].set_title("Original")
        axs[1].imshow(seg, cmap="gray"); axs[1].set_title("Segmentada")
        plt.show()

        nombre = input("Nombre para guardar segmentada (.png): ")
        cv2.imwrite(nombre, seg)
    def morfologica(self):
        z = int(input(f"Ingrese índice de corte (0-{self.shape[0]-1}): "))
        slice_2d = self.image_3d[z, :, :].astype(float)
        img_norm = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min()) * 255
        img_uint8 = img_norm.astype(np.uint8)

        k = int(input("Tamaño kernel (impar): "))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        
        print("1:Dilatación 2:Erosión 3:Apertura 4:Cierre")
        op = int(input("Operación: "))
        ops = {1: cv2.MORPH_DILATE, 2: cv2.MORPH_ERODE, 3: cv2.MORPH_OPEN, 4: cv2.MORPH_CLOSE}
        
        result = cv2.morphologyEx(img_uint8, ops[op], kernel)

        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(img_uint8, cmap="gray"); axs[0].set_title("Original")
        axs[1].imshow(result, cmap="gray"); axs[1].set_title("Morfología")
        plt.show()

        nombre = input("Nombre para guardar (.png): ")
        cv2.imwrite(nombre, result)