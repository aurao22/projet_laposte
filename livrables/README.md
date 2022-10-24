# Projet La Poste

La Poste vous mandate pour développer un prototype qui servira de POC sur la reconnaissance de chiffre.

L'idée est de créer un système qui permet de reconnaître le code postal manuscrit sur une lettre.

Pour ce faire vous montrerez votre capacité à reconnaitre un chiffre dessiné à la souris par n'importe quel utilisateur. 

Les grandes étapes du projet que vous pouvez considérer (temps et difficulté croissants):
- Sauvegarder et restaurer un modèle entraîné pour l'utiliser sur une image connue
- Dessiner une image sur un logiciel et la traiter pour la transformer au format utilisable par notre réseau
- Faire un programme python qui gère la totalité, de l'interface à l'inférence

## Quelques références que j'ai utilisées
### Interface (une option possible)
- [Paint sous python](https://gist.github.com/nikhilkumarsingh/85501ee2c3d8c0cfa9d1a27be5781f06)
- [Initiation à Tkinter](http://www.emmanuelmorand.net/programmation/LogicielDeDessinEnPython/LogicielDeDessinEnPython.pdf)

### Sauvegarde de l'image

https://www.semicolonworld.com/question/55284/how-can-i-convert-canvas-content-to-an-image


## Livrables
- Generation du modèle : laposte_creation_modele.ipynb (+laposte_model_creation.py)
- Lancement du programme de paint et prédiction : laposte_paint.py


