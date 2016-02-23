# Lecture des données

1. ouvrir le fichier *.fit
2. parcourir le XML et mettre les valeurs de puissances dans un tableau

# calcul du profil de puissance

1. Initialisation

... Créer un tableau destination avec des zéros pour t variant entre 0 et 300 minutes avec un pas d'une seconde

2. Pour tous les fichiers fit 

..1 Pour toutes les durées entre 0 et 300 sec possibles 
...1 Calculer les moyennes glissantes de la puissance
...2 Conserver le maximum
..2 Comparer les valeurs du tableau de la sortie avec le RPP et rempacer par les valeurs en corps ssi les valeurs sont supérieur


