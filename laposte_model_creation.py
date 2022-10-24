# %% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras import layers, Sequential, Input
from os import listdir
from os.path import isfile, join
import pandas as pd
# ----------------------------------------------------------------------------------
#                        DATA PRE-PROCESSING
# ----------------------------------------------------------------------------------
# %% preprocess_data
def preprocess_data(x, y,nb_classes, verbose=0):
    short_name = 'preprocessing'
    if verbose > 0:
        print(f"[{short_name}]\tINFO : x = {x.shape}, y = {y.shape} receive")

    # Standardisation des données
    x_preprocess = x.copy() / 255.0
    
    # Make sure images have shape (28, 28, 1)
    x_preprocess = np.expand_dims(x_preprocess, -1)
    
    # OneHot encoder
    # convert class vectors to binary class matrices
    y_preprocess = to_categorical(y.copy(), nb_classes)

    if verbose > 0:
        print(f"[{short_name}]\tINFO : x = {x_preprocess.shape}, y = {y_preprocess.shape} after preprocess")

    return x_preprocess, y_preprocess

# %% get_dir_files
def get_dir_files(dir_path, endwith=None, verbose=0):
    fichiers = None
    if endwith is not None:
        fichiers = [f for f in listdir(dir_path) if isfile(join(dir_path, f)) and f.endswith(endwith)]
    else:
        fichiers = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    return fichiers
# ----------------------------------------------------------------------------------
#                        MODEL
# ----------------------------------------------------------------------------------
# %% create_model
# Function to create model, required for KerasClassifier
def create_model(input_shape, num_classes):
	# create model
	model3 = Sequential(
		[
			Input(shape=input_shape, name="entree"),
			layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
			layers.MaxPooling2D(pool_size=(2, 2)),
			layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
			layers.MaxPooling2D(pool_size=(2, 2)),
			layers.Flatten(),
			layers.Dropout(0.5),
			layers.Dense(num_classes, activation="softmax"),
		]
	)
	print(model3.summary())
	model3.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
	return model3

# ----------------------------------------------------------------------------------
# %%                       GRAPHIQUES
# ----------------------------------------------------------------------------------
PLOT_FIGURE_BAGROUNG_COLOR = 'white'
PLOT_BAGROUNG_COLOR = PLOT_FIGURE_BAGROUNG_COLOR


def color_graph_background(ligne=1, colonne=1):
    figure, axes = plt.subplots(ligne,colonne)
    figure.patch.set_facecolor(PLOT_FIGURE_BAGROUNG_COLOR)
    if isinstance(axes, np.ndarray):
        for axe in axes:
            # Traitement des figures avec plusieurs lignes
            if isinstance(axe, np.ndarray):
                for ae in axe:
                    ae.set_facecolor(PLOT_BAGROUNG_COLOR)
            else:
                axe.set_facecolor(PLOT_BAGROUNG_COLOR)
    else:
        axes.set_facecolor(PLOT_BAGROUNG_COLOR)
    return figure, axes

# %% plot_history
def plot_history(history, loss_name='loss', precision='accuracy', loss_val_name=None, precision_val=None):

    plt.figure(figsize=(12,4))
    plt.subplot(121)
    
    # Fonction de coût
    plt.plot(history[loss_name], c='steelblue', label='train loss')
    if loss_val_name is not None:
        plt.plot(history[precision_val], c='coral', label='validation loss')
    plt.xlabel("Epochs")
    plt.xticks(range(0, len(history[loss_name])))
    plt.title('Fonction de coût')
    plt.legend(loc='best')
    
    # Précision
    plt.subplot(122)
    plt.plot(history[precision], c='steelblue', label='train accuracy')
    if precision_val is not None:
        plt.plot(history[precision_val], c='coral', label='validation accuracy')
    plt.xlabel("Epochs")
    plt.xticks(range(0, len(history[loss_name])))
    plt.ylabel("rate")
    plt.title('Précision')
    plt.legend(loc='best')
    plt.show()

# %% plot_pred
def plot_pred(x, y, predictions, range=range(0,1)):
    for i in range:
        plt.figure(figsize=(6,3))
        plt.subplot(1,2,1)
        _plot_image(i, predictions[i], y, x)
        plt.subplot(1,2,2)
        _plot_value_array(i, predictions[i],  y)
        plt.show()

def plot_pred_multiple(x, y, predictions, range=range(0,1)):
    # plot some of the numbers
    nb = range.stop * 2

    nb_cols = 10
    nb_lignes = (nb//nb_cols)
    if nb_lignes < 1:
        nb_lignes = 1

    plt.figure(figsize=(20,(nb_lignes*1.5)))
    tot = 1
    for i in range:
        try:
            plt.subplot(nb_lignes,nb_cols,tot)                       
            _plot_image(i, predictions[i], y, x)
            tot += 1
            plt.subplot(nb_lignes,nb_cols,tot)
            _plot_value_array(i, predictions[i],  y)
            tot += 1
        except Exception as error:
            print(f"[plot_pred_multiple] \nERROR on {i} image : {error}")

    plt.show()

# %% _plot_image
def _plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(np.round(predictions_array, 2))
  true_label_i = np.argmax(np.round(true_label, 2))
  color = 'red'
  if predicted_label == true_label_i:
    color = 'green'
  
  plt.xlabel("Expected {}, Predict : {}".format(true_label_i,predicted_label), color=color)

# %% _plot_value_array
def _plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])

  predictions = np.round(predictions_array, 2)

  thisplot = plt.bar(range(len(predictions)), predictions, color="#777777")
  plt.ylim([0, 1])
  
  predicted_label = np.argmax(np.round(predictions_array, 2))
  true_label_i = np.argmax(np.round(true_label, 2))
  color = 'red'
  if predicted_label == true_label_i:
    color = 'green'

  thisplot[true_label_i].set_color('blue')
  thisplot[predicted_label].set_color(color)

# ----------------------------------------------------------------------------------
#                        PICTURES
# ----------------------------------------------------------------------------------

def contraste_img(arr, verbose=0):
    """
    
    """
    # donc appliquer proportionnellement aux autres valeurs
    max_val = arr.max()
    if verbose>0:
        print(f"Image max = {max_val} for 255")
    res = arr.copy()
    res = res * 255
    res = res / max_val
    res = np.rint(res)
    res = res.astype(int)
    return res

def show_digit(some_digit, y):
    some_digit_image = some_digit.reshape(28, 28)
    color_graph_background(1,1)
    plt.imshow(some_digit_image, interpolation = "none", cmap = "afmhot")
    plt.title(y)
    plt.axis("off")
    plt.show()

def draw_digits(df, y=None, nb=None):
    
    # plot some of the numbers
    if nb is None:
        nb = df.shape[0]

    nb_cols = 10
    nb_lignes = (nb//nb_cols)
    if nb_lignes < 1:
        nb_lignes = 1

    plt.figure(figsize=(14,(nb_lignes*1.5)))
    for digit_num in range(0,nb):
        try:
            plt.subplot(nb_lignes,nb_cols,digit_num+1)
            grid_data = df.iloc[digit_num].values.reshape(28,28)  # reshape from 1d to 2d pixel array
            plt.imshow(grid_data, interpolation = "none", cmap = "afmhot")
            if y is not None:
                if isinstance(y, pd.DataFrame):
                    plt.title(y.iloc[digit_num])
                else:
                    plt.title(y[digit_num])
            plt.axis("off")
        except Exception as error:
            print(f"[draw_digits] \nERROR on {digit_num} image : {error}")
    plt.tight_layout()
    plt.show()
# ----------------------------------------------------------------------------------
#                        TEST
# ----------------------------------------------------------------------------------

# %% test
def _test_plot_history(verbose=1):
    history = {'loss': [0.3633415699005127,
                        0.1389123797416687,
                        0.09630565345287323,
                        0.0772591158747673,
                        0.06523780524730682],
                        'accuracy': [0.899925947189331,
                        0.9598888754844666,
                        0.9728703498840332,
                        0.978092610836029,
                        0.9808148145675659],
                        'val_loss': [0.15051931142807007,
                        0.10131336748600006,
                        0.08253464102745056,
                        0.07074826210737228,
                        0.06120727211236954],
                        'val_accuracy': [0.9610000252723694,
                        0.9728333353996277,
                        0.9796666502952576,
                        0.9819999933242798,
                        0.984666645526886]}
    plot_history(history, loss_name='loss', precision='accuracy')
    plot_history(history, loss_name='loss', precision='accuracy', loss_val_name='val_loss', precision_val='val_accuracy')

# ----------------------------------------------------------------------------------
#                        MAIN
# ----------------------------------------------------------------------------------
# %% main
if __name__ == '__main__':
    verbose = 1

    _test_plot_history(verbose=verbose)