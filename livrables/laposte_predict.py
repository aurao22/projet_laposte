# %% import
import tensorflow as tf
# Helper libraries
import numpy as np
from laposte_model_creation import preprocess_data

# load and show an image with Pillow
from PIL import Image

print("TensorFlow version:", tf.__version__)


class Laposte(object):

    IMG_SIZE = (28,28) 
    verbose = 0
    nb_classes = 10

    def __init__(self, model_save_path, verbose=0, apply_contrast=True):
        self.model_save_path = model_save_path
        self.load_model(verbose=verbose)
        self.apply_contrast = apply_contrast
                
        
    def load_model(self, verbose=0):
         # Create a basic model instance
        self.model = tf.keras.models.load_model(self.model_save_path)
        if verbose > 0:
            # Check its architecture
            print(self.model.summary())
    
    def predict(self, image, expected_label=None, verbose=0):
        # Retailler les images pour les forcer en 28x28.
        if image.size != self.IMG_SIZE:
            image = image.resize(self.IMG_SIZE)
        
        # conversion en Gris
        image = image.convert("L")
        
        # Transformation de l'image en tableau numpy
        arr = np.array(image)
        arr = arr.astype(int)
        arr = arr.reshape(image.size)
        arr = self.contrast(arr=arr, verbose=verbose)

        x_test= np.array([arr])
        y_test = None
        if expected_label is not None:
            y_test = np.array([expected_label])
        x_test, y_test = preprocess_data(x_test, y_test, nb_classes=self.nb_classes, verbose=verbose)

        one_picture_prediction = self.model.predict(x_test)
        return one_picture_prediction
        
    def convert_prediction_to_class(self, one_picture_prediction, verbose=0):
        return np.argmax(np.round(one_picture_prediction, 2))

    def contrast(self, arr, verbose=0):
        arr2 = arr.copy()
        if self.apply_contrast:
            max_val = arr2.max()
            if verbose>0:
                print(f"Image max = {max_val} for 255")
            arr2 = arr2 * 255
            arr2 = arr2 / max_val
            arr2 = np.rint(arr2)
            arr2 = arr2.astype(int)
        return arr2

# ---------------------------------------------------------------------------------------------
#                               MAIN
# ---------------------------------------------------------------------------------------------
if __name__ == '__main__':
    from os import getcwd
    from os.path import join

    verbose = 1

    # Récupère le répertoire du programme
    file_path = getcwd() + "\\"
    if "PROJETS" not in file_path:
        file_path = join(file_path, "PROJETS")

    if "projet_laposte" not in file_path:
        file_path = join(file_path, "projet_laposte")

    source_data_path = join(file_path,"data")
    model_save_path = join(file_path,'model','my_model')

    if verbose>0:
        print(f"Execution path : {file_path}")
        print("Source path :", source_data_path)
        print(f"model_save_path : {model_save_path}")

    la_poste = Laposte(model_save_path=model_save_path, verbose=verbose)
    img_path = r'C:\Users\User\WORK\workspace-ia\PROJETS\projet_laposte\data\000-test_manuel_000-b.png'
    image = Image.open(img_path)  
    predictions = la_poste.predict(image=image, expected_label=0, verbose=verbose)
    res = la_poste.convert_prediction_to_class(predictions, verbose=verbose)
    print(f"La prédiction est : {res}")
