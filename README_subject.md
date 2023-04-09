# Évaluation de python-numérique

Lisez bien ce document jusqu'au bout ! 

## **Informations importantes**

Ce projet constitue l'évaluation de la partie python-numérique de l'ECUE PE.
Le sujet est proposé par `Laurent Lacourt`; le problème et les données sont réels. Nous vous demandons de ne pas les diffuser sur Internet.

Le sujet se trouve dans le fichier `00-eval-pe.nb.py`. Vous devez compléter ce notebook avec vos implémentations et remarques. 


**Réalisation, condition et date de rendu du projet**  
* le projet doit être réalisé **individuellement**
* il doit être rendu sous la forme d'un notebook (*vous complétez le notebook initial avec votre code et vos explications*)
* vous avez **5 semaines** pour le réaliser: il doit donc être envoyé **avant le `17 décembre minuit`**
* adressez votre `notebook` par mail à `laurent.lacourt@mines-paristech.fr`  
  veuillez utiliser votre adresse `@mines-paristech.fr` comme adresse d'émission
  
Sans vouloir le moins du monde être désagréables, nous vous prévenons que les projets non envoyés à cette date, envoyés à une mauvaise adresse, réalisés à plusieurs, identiques ou fort ressemblants (oui nous savons c'est très subjectif), ne seront pas notés et les élèves concernés ne valideront pas le module Python-numérique (donc pas PE).

**Des questions ?**  
* les questions concernant le sujet, doivent être posées à Laurent Lacourt (après avoir bien réfléchi à la question)
* les questions techniques, à vos enseignants de Python-numérique (après avoir bien réfléchi à la question)
* pour tout autre problème, contactez valerie.roy@mines-paristech.fr


**Attendus sur votre code**
* le code doit être écrit en anglais
* les commentaires et les explications peuvent rester en français 
* le code doit être propre, lisible et commenté aux endroits adéquats
* vos fonctions doivent contenir une `docstring`
* le code doit s'exécuter sans erreur


* Pour les élèves en `Python avancé` votre code doit respecter *autant que possible* la `PEP8`
* Pour les élèves en `C++`, ceux qui le veulent peuvent s'y conformer mais ce n'est pas obligatoire . 


## Installation des modules nécessaires

Afin de visualiser une partie des données qui vous sont mises à disposition pour ce projet, vous devez installer un module spécifique. Pour ne pas surcharger votre environnement de base, nous vous conseillons de créer un environnement spécifique à l'évaluation et d'y installer les modules nécessaires: 

```bash
conda create -n eval-pe python=3.9
conda activate eval-pe
pip install -r requirements.txt
jupyter notebook
```

**Dans la suite, vous devrez taper la commande suivante avant de commencer à travailler sur le sujet :**

```
conda activate eval-pe
```
