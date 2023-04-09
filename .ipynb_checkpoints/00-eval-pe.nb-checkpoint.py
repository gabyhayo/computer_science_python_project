# -*- coding: utf-8 -*-
# %% [markdown]
# # Évaluation de python-numérique
#
# ***Analyse morphologique de défauts 3D***
#
# ***

# %% [markdown]
# ## Votre identité

# %% [markdown]
# Ne touchez rien, remplacez simplement les ???
#
# Prénom: Gabriel
#
# Nom: Hayoun
#
# Langage-avancé (Python ou C++): C++
#
# Adresse mail: gabriel.hayoun@mines-paristech.fr
#
# ***

# %% [markdown]
# ## Quelques éléments de contexte et objectifs
#
# Vous allez travaillez dans ce projet sur des données réelles concernant des défauts observés dans des soudures. Ces défauts sont problématiques car ils peuvent occasionner la rupture prématurée d'une pièce, ce qui peut avoir de lourdes conséquences. De nombreux travaux actuels visent donc à caractériser la **nocivité** des défauts. La morphologie de ces défauts est un paramètre qui influe au premier ordre sur cette nocivité.
#
# Dans ce projet, vous aurez l'occasion de manipuler des données qui caractérisent la morphologie de défauts réels observés dans une soudure volontairement ratée ! Votre mission est de mener une analyse permettant de classer les défauts selon leur forme : en effet, deux défauts avec des morphologies similaires auront une nocivité comparable.

# %% [markdown]
# ### Import des librairies numériques
#
# Importez les librairies numériques utiles au projet, pour le début du projet, il s'agit de `pandas`, `numpy` et `pyplot` de `matplotlib`.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# %% [markdown]
# ## Lecture des données
#
# Les données se trouvent dans le fichier `defect_data.csv`. Ce fichier contient treize colonnes de données :
# * la première colonne est un identifiant de défaut (attention, il y a 4040 défauts mais les ids varient entre 1 et 4183) ;
# * les neuf colonnes suivantes sont des descripteurs de forme sur lesquels l'étude va se concentrer ;
# * les trois dernières colonnes sont des indicateurs mécaniques auxquels nous n'allons pas nous intéresser dans ce projet.
#
# Lisez les données dans une data-frame en ne gardant que les 9 descripteurs de forme et en indexant vos lignes par les identifiants des défauts. Ces deux opérations doivent être faites au moment de la lecture des données.
#
# Affichez la forme et les deux dernières lignes de la data-frame.

# %%
df = pd.read_csv("defect_data.csv", usecols=range(0, 10), index_col="id")
df

# %%
print(f" la forme de la data-frame est { df.shape } ")
df.tail(2)

# %% [markdown]
# ### Parenthèse sur descripteurs morphologiques
#
# **Note: cette section vous donne du contexte quant à la signification des données que vous manipulez et la façon dont elles ont été acquises. Si certains aspects vous semblent nébuleux, cela ne vous empêchera pas de finir le projet !**
#
# Vous allez manipuler dans ce projet des descripteurs morphologiques. Ces descripteurs sont ici utilisés pour caractériser des défauts, observés par tomographie aux rayons X dans des soudures liant deux pièces métalliques. La tomographie consiste à prendre un jeu de radiographies (comme chez le médecin, avec un rayonnement plus puissant) en faisant tourner la pièce entre chaque prise de vue. En appliquant un algorithme de reconstruction idoine à l'ensemble des clichés, il est possible de remonter à une image 3D des pièces scannées. Plus la zone que l'on traverse est dense plus elle est claire (comme chez le médecin : vos os apparaissent plus clair que vos muscles). Dans notre cas, le constraste entre les défauts constitués d'air et le métal est très marqué : on observe donc les défauts en noir et le métal en gris. Un défaut est donc un amas de voxels (l'équivalent des pixels pour une image 3D) noirs. Sur l'image ci-dessous, les défauts ont été extraits et sont représentés en 3D par les volumes rouges.
#
# <img src="media/defects_3D.png" width="400px">
#
# Vous voyez qu'ils sont nombreux, de taille et de forme variées. À chaque volume rouge que vous observez correspond une ligne de votre `DataFrame` qui contient les descripteurs morphologiques du-dit défaut.
#
#
# #### Descripteur $r_1$ (`radius1`)
# En notant $N$ le nombre de voxels constituant le défaut, on obtient le volume du défaut $V=N\times v_0$ (où $v_0$ est le volume d'un voxel).On peut alors définir son *rayon équivalent* comme le rayon de la sphère de même volume soit :
# \begin{equation*}
#  R_{eq} = \left(\frac{3V}{4\pi}\right)^{1/3}
# \end{equation*}
#
# On définit ensuite le *rayon moyen* $R_m$ du défaut comme la moyenne sur tous les voxels de la distance au centre de gravité du défaut.
#
# $R_{eq}$ et $R_m$ portent une information sur la taille du défaut. En les combinant comme suit:
# \begin{equation*}
#  r_1 = \frac{R_{eq} - R_m}{R_m}
# \end{equation*}
# on la fait disparaître : $r_1$ vaut 1/3 pour une sphère quel que soit son rayon.
#
# **Note :** vous aurez remarqué que $r_1$ est donc sans dimension.
#
# #### Descripteurs basés sur la matrice d'inertie ($\lambda_1$ et $\lambda_2$) (`lambda1`, `lambda2`)
# La matrice d'inertie de chaque défaut est calculée. Pour ce faire, on remplace tout simplement les intégrales sur le volume présentes dans les formules que vous connaissez par une somme sur les voxels. Par exemple:
# \begin{equation}
#  I_{xy} = -\sum\limits_{v\in\rm{defect}} (x(v)-\bar{x})(y(v)-\bar{y})\qquad \text{avec } \bar{x} = \frac{1}{N}\sum\limits_{v\in\rm{defect}} x(v) \text{ et } \bar{y} = \frac{1}{N}\sum\limits_{v\in\rm{defect}} y(v)
# \end{equation}
# Cette matrice est symétrique réelle, elle peut donc être diagonalisée. Les trois valeurs propres obtenues $I_1 \geq I_2 \geq I_3$ sont les moments d'inertie du défaut dans son repère principal d'inertie. Ces derniers portent de manière intrinsèque une information sur le volume du défaut. Pour la faire disparaître, il suffit de normaliser ces moments comme suit :
#
# \begin{equation}
#  \lambda_i = \frac{I_i}{I_1+I_2+I_3}
# \end{equation}
#
# On obtient alors trois indicateurs $\lambda_1 \geq \lambda_2 \geq \lambda_3$ vérifiant notamment $\lambda_1 + \lambda_2 + \lambda_3 = 1$ ce qui explique que l'on ne garde que les deux premiers. En utilisant les propriétés des moments d'inertie, on peut montrer que les points obtenus se situent dans le triangle formé par $(1/3, 1/3)$, $(1/2, 1/4)$ et $(1/2, 1/2)$ dans le plan $(\lambda_1, \lambda_2)$. Vous pourrez vérifier cela dans la suite !
#
# La position du point dans le triangle renseigne sur sa forme *globale*, comme indiqué par l'image suivante :
# ∑
# <img src="media/l1_l2.png" width="400px">
#
# #### Convexité (`convexity`)
#
# L'indicateur de convexité utilisé est simplement le rapport entre le volume du défaut et de son convexe englobant. $C = V/V_{CH} \leq 1$. Lorsque qu'un défaut est convexe, il est égal à son convexe englobant et donc $C$ vaut 1.
#
# #### Sphéricité (`sphericity`)
#
# L'indicateur de sphéricité permet de calculer l'écart d'un défaut à une sphère. On utilise ici la caractéristique de la sphère qui minimise la surface extérieure pour un volume donné. La grandeur :
# \begin{equation*}
# I_S = \frac{6\sqrt{\pi}V}{S^{3/2}}
# \end{equation*}
# où $V$ est le volume du défaut et $S$ sa surface vaut 1 pour une sphère et est inférieur à 1 sinon.
#
# #### Indicateurs basés sur la mesure de la courbure (`varCurv`, `intCurv`)
#
# Les deux courbures principales $\kappa_1$ et $\kappa_2$ sont calculées en chaque point de la surface des défauts ([ici pour les plus curieux](https://fr.wikipedia.org/wiki/Courbure_principale)). Ces courbures permettent de caractériser la forme locale du défaut. Elle sont combinées pour calculer la courbure moyenne $H = (\kappa_1+\kappa_2)/2$ et la courbure de Gauss $\gamma = \kappa_1\kappa_2$. Pour s'affranchir de l'information sur la taille (pour une sphère de rayon $R$, on a en tout point $\kappa_1 = \kappa_2 = 1/R$), les défauts sont normalisés en volume avant d'en calculer les courbures.
# Les indicateurs retenus sont les suivants:
#
#  - la variance de la courbure de Gauss (colonne `varCurv`) $Var(H)$ ;
#  - l'intégrale de $\gamma$ sur la surface du défaut(colonne `intCurv`) $\int_S \gamma dS$.
#
# Ces indicateurs valent respectivement $0$ et $4\pi$ pour une sphère.
#
# #### Indicateurs sur la boite englobante $(\beta_1, \beta_2)$ (`b1`, `b2`)
#
# Finalement, c'est une information sur la boite englobante du défaut dans son repère principal d'inertie qui est cachée dans $(\beta_1, \beta_2)$. En notant $B_1, B_2, B_3$ les dimensions (décroissantes) de la boite englobante, on réalise la même normalisation que pour les moments d'inertie :
# \begin{equation}
#  \beta_i = \frac{B_i}{B_1+B_2+B_3}
# \end{equation}
#
# ***

# %% [markdown]
# ## Visualisation des défauts
#
# Pour que vous saissiez un peu mieux la signification des descripteurs morphologiques, nous avons concocté une petite fonction utilitaire qui vous permettra de visualiser les défauts. Pour que vous puissiez interagir avec `pyplot`, il nous est imposé de changer le backend avec la commande `%matplotlib notebook` et de recharger le module. Pour revenir dans le mode de visualisation précédent, vous devrez évaluer la cellule qui contient la commande `%matplotlib inline` qui arrive un peu plus tard !
#
# *Nous n'avons malheureusement pas trouvé de solution pour que ce changement soit transparent pour vous... :(*
#
# Amusez vous à chercher des défauts **extrêmes** pour comprendre. Par exemple le défaut qui maximise $\lambda_2$ sera celui qui a la forme la plus *filaire* alors que celui qui minimise aura la forme la plus *plate*. Pourquoi ne pas jeter un coup d'oeil au défaut le moins convexe ?

# %%
# Ne touchez pas à ça c'est pour la visualisation interactive !
# %matplotlib notebook
from importlib import reload
from utilities import plot_defect

reload(plt)
# À partir de maintenant vous pouvez vous amuser !

# On récupère un id intéressant
id_to_plot = df.index[df["convexity"].argmin()]
# On affiche à l'écran les valeurs de ses descripteurs
print(df.loc[id_to_plot])
# On l'affiche
plot_defect(id_to_plot)

# %% [markdown]
# N'oubliez pas d'aller rendre visite au défaut avec l'id `4022` qui a une forme rigolote, avec ses petites excroissances.

# %%
# Le défaut 4022 !
print(df.loc[4022])
plot_defect(4022)


# %% [markdown]
# On vous parlait juste avant de défauts de morphologie proche ! Et si une simple distance euclidienne en dimension 9 fonctionnait ? Calculez le défaut le plus proche du défaut `4022` dans l'espace de dimension 9, et tracez-le ! Se ressemblent-ils ?

# %%
def distance(x, id1, id2):
    """ arguments : une Pandas dataframe, 2 index de celle-ci
    sortie : distance euclidienne entre les 2 lignes de la dataframe dont les index sont en argument """
    return np.linalg.norm(x.loc[id1] - x.loc[id2])


x = min([(distance(df, 4022, i), i) for i in df.drop(4022).index])
# car on prend le minimum de la première composante de chaque couple
print(f" l'index de l'élément le plus proche de l'élement 4022 est : { x[ 1 ] } ")
plot_defect(x[1])

# %% [markdown]
# **Eh non!** Le défaut le plus proche du défaut `4022` est une patatoïde quelconque. Deux explications sont possibles :
#  * soit la distance euclidienne n'est pas pertinente ici ;
#  * soit le défaut `4022` est le seul avec des petites excroissances...
# Je vous laisse aller voir le défaut `796` pour trancher entre les deux propositions..
#

# %%
print(df.loc[796])
plot_defect(796)

# %% [markdown]
# le défaut 796 possède également des excroissances, ainsi il semble que la distance euclidienne ne soit pas pertinente ici.

# %%
# On revient en mode de visu standard après avoir évalué cette cellule
# %matplotlib inline
reload(plt)


# %% [markdown]
# ## Visualisation des données
#
# Avant de commencer toute analyse à proprement parler de données, il est nécessaire de passer un moment à les observer.
#
# Pour ce faire, nous allons écrire des fonctions utilitaires qui se chargeront de tracer des graphes classiques.

# %% [markdown]
# ### Tracé d'un histogramme
#
# Écrire une fonction `histogram` qui trace l'histogramme d'une série de points.
#
#
# Par exemple l'appel `histogram(df['radius1'], nbins=10)` devrait renvoyer quelque chose qui ressemble à ceci:
#
# <img src="media/defects-histogram.png" width="400px">

# %%
# votre code ici
def histogram(x, nbins=10):
    """arguments : une data-frame ou series panda et un entier nbins déterminant la largeur d'intervalles
    sortie : un histogramme """
    x.hist(bins=nbins)


# %%
# pour vérifier
histogram(df["radius1"], nbins=10)

# %%
# pour vérifier
# c'est bien si votre fonction marche aussi avec une dataframe
histogram(df[["radius1"]], nbins=10)


# %% [markdown]
# #### *Bonus* : un histogramme adaptable aux goûts de chacun
#
# Modifier la fonction `histogram` pour que l'utilisateur puisse préciser par exemple: le nom des étiquettes des axes, les couleurs à utiliser pour représenter l'histogramme...
#
# Par exemple en appelant
# ```python
# histogram2(df['radius1'], nbins=10,
#        xlabel='radius1', ylabel='occurrences',
#        histkwargs={'color': 'red'})
# ```
#
# on obtiendrait quelquechose comme ceci
#
# <img src="media/defects-histogram2.png" width="400px">

# %%
# votre code ici
def histogram2(x, nbins=10, xlabel=None, ylabel=None, histkwargs={"color": "blue"}):
    """ arguments : une data-frame ou series panda, un entier nbins déterminant la largeur des intervalles,
    les légendes des axes x et y et la couleur de l'histogramme
    sortie : un histogramme """
    x.hist(bins=nbins, **histkwargs)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


# %%
# pour vérifier

# seulement si la fonction est définie
if "histogram2" in globals():
    histogram2(
        df["radius1"],
        nbins=10,
        xlabel="radius1",
        ylabel="occurrences",
        histkwargs={"color": "red"},
    )
else:
    print("vous avez choisi de ne pas faire le bonus")


# %% [markdown]
# ### Tracé de nuages de points
#
# Écrire une fonction `correlation_plot` qui permet de tracer le nuage de points entre deux séries de données.
# L'appel de cette fontion comme suit `correlation_plot(df['lambda1'], df['lambda2'])` devrait donner une image ressemblant à celle-ci :
#
# Ces tracés illustrent le *degré de similarité* des colonnes. Notons, qu'il existe des indices de similarité comme par exemple: la covariance, le coefficient de corrélation de Pearson...
#
#
# <img src="media/defects-correlation.png" width="400px">

# %%
# votre code ici
def correlation_plot(x, y):
    """ arguments : 2 series panda x et y
    sortie : un nuage de point y en fonction de x """
    plt.scatter(x, y)


# %%
# pour vérifier
correlation_plot(df["lambda1"], df["lambda2"])


# %% [markdown]
# #### *Bonus* les nuages de points pour l'utilisateur casse-pieds (ou daltonien ;) )
# Modifier la fonction `correlation_plot` pour que l'utilisateur puisse préciser des noms pour les axes, et choisir l'aspect des points tracés (couleur, taille, forme, ...).
#
# par exemple en appelant
# ```python
# correlation_plot2(df['lambda1'], df['lambda2'],
#                   xlabel='lambda1', ylabel='lambda2',
#                   plot_kwargs={'marker': 'x', 'color': 'red'})
# ```
# on obtiendrait quelque chose comme
#
# <img src="media/defects-correlation2.png" width="400px">

# %%
# votre code ici
def correlation_plot2(
    x, y, xlabel=None, ylabel=None, plot_kwargs={"marker": "o", "color": "blue"}
):
    """ arguments : 2 series panda x et y,les légendes des axes x et y, 
    le type de marqueur et la couleur de l'histogramme
    sortie : un nuage de point y en fonction de x """
    plt.scatter(x, y, **plot_kwargs)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


# %%
# pour vérifier

# seulement si la fonction est définie
if "correlation_plot2" in globals():
    correlation_plot2(
        df["lambda1"],
        df["lambda2"],
        xlabel="lambda1",
        ylabel="lambda2",
        plot_kwargs={"marker": "x", "color": "red"},
    )
else:
    print("vous avez choisi de ne pas faire le bonus")


# %% [markdown]
# #### *Bonus 2* Tracer le triangle d'inertie en plus
#
# On vous disait plus tôt que les points dans le plan $(\lambda_1, \lambda_2)$ sont forcément dans le triangle formé par $(1/3, 1/3)$, $(1/2, 1/4)$ et $(1/2, 1/2)$.
# Essayez de superposer les données au triangle pour mettre cela en évidence !
# Le résultat pourrait ressembler à ceci :
#
# <img src="media/defects-correlation3.png" width="400px">
#
# (Vous pouvez faire ce bonus sans avoir fait le précédent)

# %%
def inertia_triangle(graphe):
    """ argument : un graphique sur lequel on souhaite affiché le triangle d'inertie
    sortie : le graphique avec le triagnle d'inertie superposé"""
    x1 = np.linspace(1 / 3, 1 / 2)
    x3 = np.full((50,), 1 / 2)
    y1 = np.linspace(1 / 3, 1 / 4)
    y2 = np.linspace(1 / 3, 1 / 2)
    y3 = np.linspace(1 / 4, 1 / 2)
    plt.plot(x1, y1, c="r", linestyle="dashed")
    plt.plot(x1, y2, c="r", linestyle="dashed")
    plt.plot(x3, y3, c="r", linestyle="dashed")
    graphe


# %%
inertia_triangle(
    correlation_plot2(
        df["lambda1"],
        df["lambda2"],
        xlabel="lambda1",
        ylabel="lambda2",
        plot_kwargs={"marker": "x", "color": "black"},
    )
)


# %% [markdown]
# ### Affichage de tous les plots des colonnes
#
# Écrire une fonction `plot2D` qui prend en argument une dataframe et qui affiche
# * les histogrammes des colonnes
# * les plots des corrélations des couples des colonnes
# n'affichez qu'une seule fois les corrélations par couple de colonnes
#
# avec `plot2D(df[['radius1', 'lambda1', 'lambda2']])` vous devriez obtenir quelque chose comme ceci
#
# <img src="media/defects-plot2d.png" width="200px">

# %%
# votre code ici
def plot2D(df):
    """argument : une Panda dataframe
    sortie : les histogrammes des colonnes puis les corrélations des couples de colonnes (nuage de point)"""
    nb_col = df.shape[1]
    col = df.columns
    plt.figure(1, figsize=(10, 20))
    plt.gcf().subplots_adjust(hspace=0.5)
    for i in range(nb_col):
        plt.subplot(2 * nb_col, 1, i + 1)
        histogram2(df.iloc[:, i], nbins=50, xlabel=col[i], ylabel="Frequency")
        plt.subplot(2 * nb_col, 1, i + nb_col + 1)
        correlation_plot2(
            df.iloc[:, i - 1],
            df.iloc[:, i],
            xlabel=col[i - 1],
            ylabel=col[i],
            plot_kwargs={"marker": "o", "color": "black"},
        )


# %%
# pour corriger

plot2D(df[["radius1", "lambda1", "lambda2"]])


# %% [markdown]
# #### *Bonus++* (dataviz expert-level) le tableau des plots des colonnes
#
# Écrire une fonction `scatter_matrix` qui prend une dataframe en argument et affiche un tableau de graphes avec
# * sur la diagonale les histogrammes des colonnes
# * dans les autres positions les plots des corrélations des couples des colonnes
#
# avec `scatter_matrix(df[['radius1', 'lambda1', 'b2']], nbins=100, hist_kwargs={'fc': 'g'})` vous devriez obtenir à peu près ceci
#
# <img src="media/defects-matrix.png" width="500px">

# %%
def scatter_matrix(df, nbins=100, histkwargs={"fc": "g"}):
    """ arguemtns : une Panda dataframe, un entier nbins déterminant la largeur des intervalles,
    la couleur des histogrammes
    sortie : une matrice carré de taille le nombre de colonne de la dataframe, sur sa diagonale il y a
    les histogrammes des colonnes de la dataframe, et dans les autres positions les plots des corrélations 
    des couples des colonnes."""
    nb_col = df.shape[1]
    col = df.columns
    lst = [1 + i * (1 + nb_col) for i in range(nb_col)]
    param1 = np.concatenate((lst, np.setdiff1d(range(2, nb_col ** 2), lst)))
    param2 = np.array(range(1, nb_col))
    for i in range(1, nb_col):
        param2 = np.concatenate((param2, np.setdiff1d(range(0, nb_col), [i])))
    var = nb_col
    elem = 0
    plt.figure(1, figsize=(25, 40))
    plt.gcf().subplots_adjust(
        left=0.5, bottom=0.5, right=1.5, top=0.9, wspace=0.5, hspace=0.5
    )
    for i in range(nb_col):
        plt.subplot(nb_col, nb_col, param1[i])
        histogram2(
            df.iloc[:, i],
            nbins=50,
            xlabel=col[i],
            ylabel="Frequency",
            histkwargs=histkwargs,
        )
        for j in range(nb_col - 1):
            plt.subplot(nb_col, nb_col, param1[var])
            correlation_plot2(
                df.iloc[:, param2[elem]],
                df.iloc[:, i],
                xlabel=col[param2[elem]],
                ylabel=col[i],
                plot_kwargs={"marker": "o", "color": "black"},
            )
            var += 1
            elem += 1
    plt.show()


# %%
# pour vérifier

scatter_matrix(df[["radius1", "lambda1", "b2"]], nbins=100, histkwargs={"fc": "g"})

# %% [markdown]
# ### Corrélations entre les données
#
# Utilisez les fonctions que vous venez d'implémenter pour rechercher (visuellement) les meilleures correlations qui ressortent entre les différentes caractéristiques morphologiques.
#
# Plottez la corrélation qui vous semble la plus frappante (i.e. la plus informative), motivez votre choix.
#

# %% [markdown]
# les 2 cellules les plus corrélées seront celles dont les nuages de points se rapprocheront le plus d'une droit passant par l'origine.

# %%
# votre code ici
scatter_matrix(df, nbins=100, histkwargs={"fc": "r"})


# %% [markdown]
# les 3 graphiques qui se rapprochent le plus d'une droite sont sphericity en fonction de convexity, b1 en fonction lambda1, b2 en fonction de lambda2

# %%
plt.figure(1, figsize=(10, 5))
plt.gcf().subplots_adjust(hspace=10, right=1.5)
plt.subplot(1, 3, 1)
correlation_plot2(
    df["convexity"], df["sphericity"], xlabel="convexity", ylabel="sphericity"
)
plt.subplot(1, 3, 2)
correlation_plot2(df["lambda1"], df["b1"], xlabel="lambda1", ylabel="b1")
plt.subplot(1, 3, 3)
correlation_plot2(df["lambda2"], df["b2"], xlabel="lambda2", ylabel="b2")

# %%
x = np.polyfit(df["convexity"], df["sphericity"], 0, full=True)[1][0]
y = np.polyfit(df["lambda1"], df["b1"], 0, full=True)[1][0]
z = np.polyfit(df["lambda2"], df["b2"], 0, full=True)[1][0]
print(x, y, z)

# %% [markdown]
# On remarque le coefficient caractérisant l'approximation par une droite (les résidus) est le plus proche de 0 pour b2 en fonction de lambda2.

# %% [markdown]
# ## Analyse en composantes principales (ACP)
#
# Les corrélations entre variables mises en évidence précédemment nous indiquent que certaines informations, apportées par les indicateurs, sont parfois redondantes.
#
# L'analyse en composantes principales est une méthode qui va permettre de construire un jeu de *composantes principales* qui sont des combinaisons linéaires des caratéristiques. Ces composantes principales sont indépendantes les unes des autres. Ce type d'analyse est généralement mené pour réduire la dimension d'un problème.
#
# La construction des composantes principales repose sur l'analyse aux valeurs propres d'une matrice indiquant les niveaux de corrélations entre les caractéristiques. En notant $X$ la matrice contenant nos données qui est donc constituée ici de 4040 lignes et 9 colonnes, la matrice de corrélation des caractéristiques demandée ici est $C = X^TX$.

# %% [markdown]
# ### Construction des composantes principales sur les caractéristiques morphologiques
#
# Construisez une matrice des niveaux de corrélation des caractéristiques. Elle doit être carrée de taille 9x9.

# %%
# votre code ici
X = df.to_numpy()
C = np.dot(X.T, X)
print(
    f" on a construit la matrice des niveaux de corrélation C de dimension { np.shape(C) } "
)

# %% [markdown]
# ### Calcul des vecteur propres et valeurs propres de la matrice de corrélation

# %% [markdown]
# Calculez à l'aide du module `numpy.linalg` les valeurs propres et les vecteurs propres de la matrice $C$. Cette dernière est symétrique définie positive par construction, toutes ses valeurs propres sont strictement positives.

# %%
# votre code ici
eigvalues, eigvectors = np.linalg.eig(C)
eigvectors = (
    eigvectors.T
)  # car le vecteur propre correspondant à la valeur propre i est la i-ème colonne
print(
    f" La liste des valeurs propres de C est { eigvalues }\nCelle des vecteurs propres est { eigvectors } "
)

# %% [markdown]
# ### Tracé des valeurs propres

# %% [markdown]
# Tracez les différentes valeurs propres calculées en utilisant un axe logarithmique.

# %%
# votre code ici
plt.plot([i + 1 for i in range(9)], eigvalues, linestyle="none", marker="o")
plt.yscale("log")
plt.show()


# %% [markdown]
# ### Analyse de l'importance relative des composantes principales

# %% [markdown]
# Vous devriez constater que les valeurs propres décroissent vite. Cette décroissance traduit l'importance relative des composantes principales.
#
# Dans le cas où les valeurs propres sont ordonnées de la plus grande à la plus petite ($\forall (i,j) \in \{1, \ldots, N\}^2, i>j
#  \Rightarrow \lambda_i \leq \lambda_j$), tracez l'évolution de la quantité suivante:
# \begin{equation*}
#  \alpha_i = \frac{\sum\limits_{j=1}^{j=i}\lambda_j}{\sum\limits_{j=1}^{j=N}\lambda_j}
# \end{equation*}
#
# $\alpha_i$ peut être interprété comme *la part d'information du jeu de données initial contenu dans les $i$ premières composantes principales*.

# %%
# votre code ici
# on remarque que eigvalues contient les valeurs propres dans l'ordre décroissant.


def alpha(i, lst):
    """ argument : un entier i compris entre 1 et 9, et une liste triée dans l'ordre décroissante
    sortie : le coeffcient alpha_i définit ci-dessus """
    tot_sum = 0
    part_sum = 0
    for k in range(9):
        tot_sum += lst[k]
        if k <= i - 1:
            part_sum += lst[k]
    return part_sum / tot_sum


# %%
eigvalues_dec = np.sort(eigvalues)[::-1]
lst_alpha = [alpha(i, eigvalues_dec) for i in range(1, 10)]
plt.plot([i + 1 for i in range(9)], lst_alpha, linestyle="none", marker="o")
plt.show()

# %% [markdown]
# ### Quantité d'information contenue par la plus grande composante principale
#
# Affichez la plus grande valeur propre et le vecteur propre correspondant (ça doit correspondre à la première composante principale).
#
# Quelle est la quantité d'information contenue par cette composante ?
#
# Pratiquement toute l'information ! C'est trop beau pour être vrai non ?
#
# Affichez les coefficients de cette composante principale. Que remarquez vous ? (*hint* cherchez la caractéristique dont le coefficient est le plus important en valeur absolue)
#
# En observant les données correspondant à cette caractéristique, avez-vous une idée de ce qui s'est passé ?

# %%
# votre code ici
max_eigvalues = max(eigvalues)
max_ind = np.where(eigvalues == max_eigvalues)[0][0]
print(
    f"la plus grande valeur propre est {max_eigvalues}\nson vecteur propre associé est {eigvectors[max_ind]}"
)

# %%
max_ind_dec = np.where(eigvalues_dec == max_eigvalues)[0][0]
# pour avoir le bon indice dans lst_alpha dont les indices correspondent à ceux des valeurs propres triées
# dans l'ordre croissant
print(
    f"la quantité d'information contenue par cette composante est {lst_alpha[max_ind_dec] :.3f}"
)

# %%
print(f"les coefficents de cette composante principale sont :\n{eigvectors[max_ind]}")

# %%
max_coeff = max(abs(eigvectors[max_ind]))
max_ind_coeff = np.where(abs(eigvectors[max_ind]) == max_coeff)[0][0]

print(
    f"Le coefficient maximal est {max_coeff}\nLa caractéristique correspondante est {df.columns[max_ind_coeff]}"
)


# %% [markdown]
# Si on regarde les moyennes de chaque variable, on remarque que celle de intCurv est nettement supérieur aux autres (aux alentours de 10), il est donc nécessaire de standardisé sinon la variable intCurv prend trop d'importance.

# %% [markdown]
# ## ACP sur les caractéristiques standardisées
#
# Dans la section précédente, la première composante principale ne prenait en compte que la caractéristique de plus grande variance. Un moyen de s'affranchir de ce problème consiste à **standardiser** les données. Pour un échantillon $Y$ de taille $N$, la variable standardisée correspondante est $Y_{std}=(Y-\bar{Y})/\sigma(Y)$ où $\bar{Y}$ est la moyenne empirique de l'échantillon et $\sigma(Y)$ son écart type empirique.
#
# **Notez que** dans notre cas, il faut réaliser la standardisation **caractéristique par caractéristique** (soit colonne par colonne). Si vous n'y avez pas encore pensé, refaites un petit tour sur le cours d'agrégation pour faire ça de manière super efficace ! ;)
#
# Menez la même étude que précédement (i.e. à partir de la section `Analyse en composantes principales`) jusqu'à tracer l'évolution des $\alpha_i$.

# %%
def standardisation(X):
    """ argument : une dataframe panda
    sortie : cette dataframe standardisé """
    X = (X - X.mean()) / X.std()
    return X


# %%
df_std = standardisation(df)
X_std = df_std.to_numpy()
C_std = np.dot(X_std.T, X_std)
print(
    f" on a construit la matrice des niveaux de corrélation C_std de dimension { np.shape(C_std) } "
)
C_std

# %%
eigvalues_std, eigvectors_std = np.linalg.eig(C_std)
eigvectors_std = eigvectors_std.T
print(
    f"La liste des valeurs propres de C_std est { eigvalues_std }\nCelle des vecteurs propres est { eigvectors_std } "
)

# %%
plt.plot([i + 1 for i in range(9)], eigvalues_std, linestyle="none", marker="o")
plt.yscale("log")
plt.show()

# %%
eigvalues_dec_std = np.sort(eigvalues_std)[::-1]
lst_alpha_std = [alpha(i, eigvalues_dec_std) for i in range(1, 10)]
plt.plot([i + 1 for i in range(9)], lst_alpha_std, linestyle="none", marker="o")
plt.show()

# %%
max_eigvalues_std = max(eigvalues_std)
max_ind_std = np.where(eigvalues_std == max_eigvalues_std)[0][0]
print(
    f"la plus grande valeur propre est {max_eigvalues_std}\nson vecteur propre associé est {eigvectors_std[max_ind_std]}"
)

# %%
max_ind_dec_std = np.where(eigvalues_dec_std == max_eigvalues_std)[0][0]
print(
    f"la quantité d'information contenue par cette composante est {lst_alpha_std[max_ind_dec_std]:.3f}"
)

# %%
print(
    f"les coefficents de cette composante principale sont :\n{eigvectors_std[max_ind_std]}"
)

# %%
max_coeff_std = max(abs(eigvectors_std[max_ind_std]))
max_ind_coeff_std = np.where(abs(eigvectors_std[max_ind_std]) == max_coeff_std)[0][0]

print(
    f"Le coefficient maximal est {max_coeff_std}\nLa caractéristique correspondante est {df.columns[max_ind_coeff_std]}"
)

# %% [markdown]
# ### Importance des composantes principales
#
# Quelle part d'information est contenue dans les 3 premières composantes principales ?
#

# %%
# votre code ici

print(
    f"La part d'information contenue dans les 3 premières composantes principales est {lst_alpha_std [2] :.3f}"
)

# %% [markdown]
# Cette part d'information est satisfaisante car nous avons tout de même réduit notre dimension de 9 à 3.

# %% [markdown]
# ### Projection dans la base des composantes principales de nos 4040 défauts
#
# On va convertir les données initiales dans la base des (vecteurs propres des) composantes principales, et les projeter sur l'espace engendré par les premiers vecteurs propres.
#
# Faites attention, vous calculez désormais dans les données standardisées.
#
# Créez une nouvelle dataframe dont les colonnes correspondent aux projections sur le sous-espace des 3 vecteurs propres prépondérants; on appellera ses colonnes P1, P2 et P3

# %%
# votre code
# df1 = np.dot(df, eigvectors_std.T)
# on a C_std qui est une matrice symétrique réelle
# on obtient ainsi une base orthonormales de vecteurs propres

P1 = np.dot(df_std, eigvectors_std[0].T)
P2 = np.dot(df_std, eigvectors_std[1].T)
P3 = np.dot(df_std, eigvectors_std[2].T)

df2 = pd.DataFrame({"P1": P1, "P2": P2, "P3": P3}, index=df.index)
df2

# %% [markdown]
# Tracez les nuages de points correspondants dans les plans (P1, P2) et (P1, P3).

# %%
plt.figure(1, figsize=(10, 15))
plt.gcf().subplots_adjust(
    left=0.125, bottom=0.4, right=1.5, top=0.9, wspace=0.5, hspace=0.2
)
plt.subplot(2, 1, 1)
correlation_plot2(
    P1, P2, xlabel="P1", ylabel="P2", plot_kwargs={"marker": "x", "color": "green"}
)
plt.subplot(2, 1, 2)
correlation_plot2(
    P1, P3, xlabel="P1", ylabel="P3", plot_kwargs={"marker": "x", "color": "orange"}
)

# %% [markdown]
# ## La conclusion
#
# Reprenez maintenant le défaut `4022` et cherchez son plus proche voisin en utilisant la distance euclidienne dans l'espace des composantes principales. Que constatez-vous ?

# %%
# %matplotlib notebook
from importlib import reload

reload(plt)
from utilities import plot_defect
from importlib import reload

# %%
# Votre code ici
y = min([(distance(df2, 4022, i), i) for i in df.drop(4022).index])
print(f" l'index de l'élément le plus proche de l'élement 4022 est : { y [ 1 ] } ")
plot_defect(y[1])


# %%
# %matplotlib inline
reload(plt)

# %% [markdown]
# **Une note de fin :** L'objectif de la démarche n'est pas seulement de trouver le plus proche voisin, mais l'ensemble des voisins (recherchez `triangulation de Delaunay` ou `tesselation de Voronoï` si vous être curieux). Or bien que les algorithmes de construction des triangulations/tessellations sont écrits en dimension quelconque, ils sont beaucoup plus efficaces en dimension faible. Dans le cas actuel, une triangulation en dimension 3 est instantanée (avec `scipy.spatial.Delaunay`) alors qu'elle met au moins une heure en dimension 9 (peut-être beaucoup plus, j'ai dû couper car mon train arrivait à destination...)

# %% [markdown]
# ***
