# %% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical

# ----------------------------------------------------------------------------------
#                        DATA PRE-PROCESSING
# ----------------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------------
#                        GRAPHIQUES
# ----------------------------------------------------------------------------------
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
        _plot_value_array(i, predictions[i],  y|i)
        plt.show()

# %% _plot_image
def _plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(predicted_label,
                                100*np.max(predictions_array),
                                true_label),
                                color=color)

# %% _plot_value_array
def _plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(len(predictions_array)), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  c = 'red'
#   if predicted_label == true_label:
#     c = 'green'

  thisplot[true_label].set_color('blue')
  thisplot[predicted_label].set_color(c)
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