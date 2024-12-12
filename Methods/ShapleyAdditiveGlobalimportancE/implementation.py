import numpy as np
import torch
import torch.nn.functional as F
import random
import cv2

class SAGEExplainer:
    def __init__(self, model, device):
        """
        Inicializa el explainer con el modelo y el dispositivo.
        model: modelo PyTorch entrenado.
        device: torch.device para ejecutar el modelo.
        """
        self.model = model
        self.device = device

    class PyTorchModelWrapper:
        def __init__(self, model, device):
            self.model = model
            self.device = device

        def predict(self, images_np):
            # Se asume (N,H,W,C) con C=1 en MNIST. Ajustar si es diferente.
            if images_np.ndim == 3:
                # (H,W,C) -> (1,C,H,W)
                images_np = images_np.transpose((2,0,1))[np.newaxis,...]
            elif images_np.ndim == 4:
                # (N,H,W,C) -> (N,C,H,W)
                images_np = images_np.transpose((0,3,1,2))

            x_t = torch.from_numpy(images_np).float().to(self.device)
            self.model.eval()
            with torch.no_grad():
                logits = self.model(x_t)  # (N, num_classes)
                probs = F.softmax(logits, dim=1).cpu().numpy()
            return probs

    @staticmethod
    def loss_fn(pred, y_true):
        return -np.log(pred[y_true] + 1e-12)

    # staticmethod
    # def segment_image_into_superpixels(image,grid_size=4):
    #     # Divide la imagen en 4 superpíxeles
    #     H, W, C = image.shape
    #     half_H, half_W = H // grid_size, W // grid_size
    #     superpixels = []
    #     # top-left
    #     mask_tl = np.zeros((H,W), dtype=bool)
    #     mask_tl[:half_H, :half_W] = True
    #     superpixels.append(mask_tl)
    #     # top-right
    #     mask_tr = np.zeros((H,W), dtype=bool)
    #     mask_tr[:half_H, half_W:] = True
    #     superpixels.append(mask_tr)
    #     # bottom-left
    #     mask_bl = np.zeros((H,W), dtype=bool)
    #     mask_bl[half_H:, :half_W] = True
    #     superpixels.append(mask_bl)
    #     # bottom-right
    #     mask_br = np.zeros((H,W), dtype=bool)
    #     mask_br[half_H:, half_W:] = True
    #     superpixels.append(mask_br)
    #     return superpixels

    @staticmethod
    def segment_image_into_superpixels(image,grid_size=4):

        H, W, C = image.shape
        step_H = H // grid_size
        step_W = W // grid_size
        superpixels = []

        for i in range(grid_size):
            for j in range(grid_size):
                mask = np.zeros((H, W), dtype=bool)
                start_H = i * step_H
                end_H = (i + 1) * step_H if i < grid_size -1 else H
                start_W = j * step_W
                end_W = (j + 1) * step_W if j < grid_size -1 else W
                mask[start_H:end_H, start_W:end_W] = True
                superpixels.append(mask)

        return superpixels

    @staticmethod
    def inpaint_with_distribution(image, known_superpixels_masks):
        im_copy = image.copy()
        H, W, C = image.shape
        known_mask = np.zeros((H, W), dtype=bool)
        for kmask in known_superpixels_masks:
            known_mask = known_mask | kmask
        mean_val = np.mean(image, axis=(0,1))
        im_copy[~known_mask] = mean_val
        return im_copy

    @staticmethod
    def model_predict(wrapper, image):
        return wrapper.predict(image)[0]

    @staticmethod
    def estimate_marginal_prediction(wrapper, X):
        preds = []
        for x in X:
            p = SAGEExplainer.model_predict(wrapper, x)
            preds.append(p)
        return np.mean(preds, axis=0)

    def explain(self, X, Y, n=10, m=5, grid_size=4):
        """
        Calcula la aproximación SAGE values para el conjunto de datos (X,Y).
        X: np.array (N,H,W,C) con las imágenes.
        Y: np.array (N,) con las etiquetas verdaderas.
        n: número de iteraciones externas.
        m: número de muestras internas.

        Retorna: np.array con los SAGE values para cada superpíxel.
        """
        self.grid_size_glob = grid_size
        # Crear el wrapper
        wrapper = self.PyTorchModelWrapper(self.model, self.device)

        # Segmentar la primera imagen para obtener superpíxeles
        superpixels_example = self.segment_image_into_superpixels(X[0],grid_size)
        d = len(superpixels_example)

        # Predicción marginal
        marginalPred = self.estimate_marginal_prediction(wrapper, X)
        phi_hat = np.zeros(d, dtype=float)
        N = len(X)

        for _ in range(n):
            idx = np.random.randint(N) #used indx but you can change _ to i
            x = X[idx]  # (H,W,C)
            y_true = Y[idx]

            perm = np.random.permutation(d)
            loss_prev = self.loss_fn(marginalPred, y_true)

            S = []
            known_sp_masks = []

            for j in range(d):
                S.append(perm[j])
                known_sp_masks = [superpixels_example[sidx] for sidx in S]

                preds = []
                for _inner in range(m):
                    inpainted = self.inpaint_with_distribution(x, known_sp_masks)
                    p = self.model_predict(wrapper, inpainted)
                    preds.append(p)
                y_bar = np.mean(preds, axis=0)

                loss_curr = self.loss_fn(y_bar, y_true)
                delta = loss_prev - loss_curr
                phi_hat[perm[j]] += delta
                loss_prev = loss_curr

        phi_hat /= n
        self.phi_array = phi_hat
        return phi_hat

    # Función para generar colores
    def get_color(self,value, min_value, max_value, positive=True):
      normalized = (value - min_value) / (max_value - min_value) if max_value != min_value else 1
      normalized = max(0, min(normalized, 1))  # Clampeo entre [0, 1]

      if positive:
          r = int(0 + (150 - 0) * (1 - normalized))
          g = int(150 + (255 - 150) * normalized)
          b = 0
      else:
          r = int(150 + (255 - 150) * normalized)
          g = int(0 + (150 - 0) * (1 - normalized))
          b = 0

      return (r, g, b)

        
    def generate_color_scale(self, phi_array):
      """
      Genera una lista de colores (tuplas RGB) según los valores de phi_array.
      Valores positivos -> Escala de verde.
      Valores negativos -> Escala de rojo.

      :param phi_array: Lista de valores.
      :return: Lista de tuplas RGB correspondientes.
      """
      #REVIEW SCALE
      # Dividir la lista en valores positivos y negativos con sus índices originales
      positive_values = [(i, value) for i, value in enumerate(phi_array) if value >= 0]
      negative_values = [(i, value) for i, value in enumerate(phi_array) if value < 0]

      # Obtener los valores separados
      positive_indices, positive_vals = zip(*positive_values) if positive_values else ([], [])
      negative_indices, negative_vals = zip(*negative_values) if negative_values else ([], [])

      # Calcular límites
      min_positive = min(positive_vals) if positive_vals else 0
      max_positive = max(positive_vals) if positive_vals else 0
      min_negative = min(negative_vals) if negative_vals else 0
      max_negative = max(negative_vals) if negative_vals else 0

      # Generar colores para cada grupo
      positive_colors = [self.get_color(value, min_positive, max_positive, positive=True) for value in positive_vals]
      negative_colors = [self.get_color(value, min_negative, max_negative, positive=False) for value in negative_vals]

      # Reconstruir la lista de colores en el orden original
      color_list = [None] * len(phi_array)
      for i, color in zip(positive_indices, positive_colors):
          color_list[i] = color
      for i, color in zip(negative_indices, negative_colors):
          color_list[i] = color

      return color_list


    def make_map(self, image):
      """
      Superpone parches de manera semi-transparente sobre una imagen de dígito MNIST, manteniendo visible el dígito original.
      
      Parámetros:
      - mnist_image: Imagen numpy array de un dígito MNIST, con dimensión (28, 28, 1)
      
      Retorna:
      - Imagen con parches superpuestos
      """

      # Verificar dimensiones de la imagen
      if image.ndim != 3 or image.shape[0] != 28 or image.shape[1] != 28 or image.shape[2] != 1:
          raise ValueError("La imagen debe ser de dimensión (28, 28, 1)")

      if image.dtype != np.uint8:
        # Convertir a uint8 si es necesario
        image = image.astype(np.uint8)
      
      # Normalizar valores de píxeles
      image = (image - image.min()) * 255 / (image.max() - image.min())
      image = image.astype(np.uint8)

      image_3ch = np.repeat(image, 3, axis=2)
      
      # Copia para modificar
      img_pathcs= image_3ch.copy()

      h_img, w_img = img_pathcs.shape[:2]

      w_patch = h_img/self.grid_size_glob
      h_patch = w_img/self.grid_size_glob

      color_array = self.generate_color_scale(self.phi_array)

      for i in range(self.grid_size_glob):
            for j in range(self.grid_size_glob):

                # Tamaños de parche
                w = w_patch
                h = h_patch
                
                # Coordenadas aleatorias
                x = i*w_patch
                y = j*h_patch

                x=int(x)
                y=int(y)
                w=int(w)
                h=int(h)          

                color = color_array[i+j]

               
                
                # Opacidad
                opacity = 0.5
                
                # Región original del parche
                region_original = img_pathcs[y:y+h, x:x+w]
                
                # Crear parche
                parche = np.full((h, w, 3), color, dtype=img_pathcs.dtype)
                
                
                # Mezclar parche
                parche_mezclado = cv2.addWeighted(region_original, 1-opacity, parche, opacity, 0)
                
                # Colocar parche
                img_pathcs[y:y+h, x:x+w] = parche_mezclado
          
      return img_pathcs
      


