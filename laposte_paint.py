from tkinter import *
from tkinter.colorchooser import askcolor
from PIL import Image
from PIL import EpsImagePlugin
from laposte_predict import *

class Paint(object):

    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = 'black'
    NB_ROW = 2
    NB_COL = 5

    def __init__(self, model_path):
        self.model_path = model_path
        self.la_poste = None
        self.root = Tk()
        self.root.title('La Poste')

        self.buttons = []

        self.pen_button = Button(self.root, text='pen', command=self.use_pen)
        self.buttons.append(self.pen_button)

        self.save_button = Button(self.root, text='save', command=self.use_save)
        self.buttons.append(self.save_button)

        self.color_button = Button(self.root, text='color', command=self.choose_color)
        self.buttons.append(self.color_button)

        self.eraser_button = Button(self.root, text='eraser', command=self.use_eraser)
        self.buttons.append(self.eraser_button)

        self.reset_button = Button(self.root, text='reset', command=self.use_reset)
        self.buttons.append(self.reset_button)

        self.choose_size_button = Scale(self.root, from_=30, to=40, orient=HORIZONTAL)
        self.buttons.append(self.choose_size_button)

        self.predict_button = Button(self.root, text='predict', command=self.use_predict)
        self.buttons.append(self.predict_button)

        self.NB_COL = len(self.buttons)

        for i in range(0, self.NB_COL):
            b = self.buttons[i]
            b.grid(row=0, column=i)

        self.etiquette=Label(self.root)
        self.etiquette.configure(text='Ma future prediction',fg='blue')
        self.etiquette.grid(row=1, column=1)

        self.c = Canvas(self.root, bg='white', width=600, height=600)
        self.c.grid(row=self.NB_ROW, columnspan=self.NB_COL)

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.pen_button
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def use_pen(self):
        self.activate_button(self.pen_button)

    def use_save(self):
        fileName=r"C:\Users\User\WORK\workspace-ia\PROJETS\projet_laposte\tmp\file_name.png"
        self.save_as_png(fileName=fileName)
        self.etiquette.configure(text='Fichier sauvegardé',fg='green')

    def use_predict(self, fileName='file_name.png', verbose=0):
        if verbose>0:
            print(f"model_save_path : {self.model_path}")

        if self.la_poste is None:
            self.la_poste = Laposte(model_save_path=self.model_path, verbose=verbose)
        image = self.draw_to_image(fileName=fileName, verbose=verbose)
        predictions = self.la_poste.predict(image=image, expected_label=0, verbose=verbose)
        res = self.la_poste.convert_prediction_to_class(predictions, verbose=verbose)
        self.etiquette.configure(text=f"La prédiction est : {res}",fg='blue')

    def choose_color(self):
        self.eraser_on = False
        self.color = askcolor(color=self.color)[1]

    def use_eraser(self):
        self.activate_button(self.eraser_button, eraser_mode=True)

    def use_reset(self):       
        self.c = Canvas(self.root, bg='white', width=600, height=600)
        self.c.grid(row=self.NB_ROW, columnspan=self.NB_COL)
        self.etiquette.configure(text='Ma future prédiction',fg='blue')
        self.setup()
        self.root.mainloop()

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
        self.line_width = self.choose_size_button.get()
        paint_color = 'white' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None

    def save_as_png(self, fileName, verbose=0):
        pic = self.draw_to_image(fileName=fileName, verbose=verbose)

        # Save to PNG
        pic.save(fileName) 

    def draw_to_image(self, fileName, verbose=0):
        # save postscipt image 
        ps_file_name = fileName.replace(".png", ".eps")
        self.c.postscript(file = ps_file_name) 

        # use PIL to convert to PNG 
        # EpsImagePlugin.gs_windows_binary =  r'X:\...\gs\gs9.52\bin\gswin64c'
        EpsImagePlugin.gs_windows_binary =  r'C:\Program Files\gs\gs10.00.0\bin\gswin64c'
        pic = Image.open(ps_file_name) 
        pic.load(scale=10)

        # Ensure scaling can anti-alias by converting 1-bit or paletted images
        if pic.mode in ('P', '1'):
            pic = pic.convert("RGB")

        # Calculate the new size, preserving the aspect ratio
        ratio = min(28 / pic.size[0],
                    28 / pic.size[1])
        new_size = (int(pic.size[0] * ratio), int(pic.size[1] * ratio))

        # Resize to fit the target size
        pic = pic.resize(new_size, Image.ANTIALIAS)
        return pic


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

    model_save_path = join(file_path,'model','my_model')

    Paint(model_path=model_save_path)
