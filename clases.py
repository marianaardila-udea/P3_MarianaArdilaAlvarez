
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