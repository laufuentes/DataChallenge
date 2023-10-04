import numpy as np
import matplotlib.pyplot as plt
import pandas 
from general import CV_rep


def f1(ytr,pred, lab): 
    """Fonction qui calcule le score f1 associé à une prédiction (pour un problème de classification binaire, C=[3,lab] avec lab={0,2})

    Args:
        ytr (pd.DataFrame): vecteur contenant les labels du jeu de données train
        pred (np.array): vecteur contenant des prédictions (2 classes)
        lab (_type_): _description_

    Returns:
        (2*TP)/(2*TP + FP + FN) (float): valeur du score f1 associé à cette prédiction
    """

    #ytr = ytr.to_numpy()
    index_true = np.where(ytr==lab)[0]
    index_false = np.where(ytr!=lab)[0]
    TP = len(np.where(pred[index_true]==lab)[0])/len(ytr)
    FN = len(np.where(pred[index_true]!=lab)[0])/len(ytr)
    FP = len(np.where(pred[index_false]==lab)[0])/len(ytr)
    return (2*TP)/(2*TP + FP + FN)

def frontiere(Xtr,ytr, lab, folds, nb_seuils): 
    """Cette fonction crée un vecteur contenant, nb_seuils, seuils différents et teste leur performance (f1 score) 
    pour classfifier le label lab. On utilisera la méthode de cross validation pour tester les performances 
    moyennes de chaque seuil.  

    Args:
        Xtr (pd.DataFrame): Jeu de données train contenant les co-variables
        ytr (pd.DataFrame): Jeu de données train contenant la variable à prédire
        lab (int: 0 ou 2): label pour lequel calculer la valeur frontière 
        folds (int): nombre de folds a créer pour performer la cross-validation
        nb_seuils (int): nombre de seuils à tester entre la valeur min,max de redshift

    Returns:
        results.mean(axis=0) (np.array): vecteur contenant les f1-scores moyennes de chaque seuil par cross-validation
    """
    seuils = np.linspace(Xtr[["redshift"]].min()[0], Xtr[["redshift"]].max()[0], nb_seuils)
    results = np.zeros((folds,nb_seuils))
    X, y = CV_rep(Xtr, ytr, folds)

    for i in range(folds): 
        Xi = X[i]
        yi=y[i].to_numpy()
        yi[np.where(yi!=lab)[0]] = 3
        for j, s in enumerate(seuils):
            pred = 3*np.ones(yi.shape[0]).reshape(-1,1)
            index = np.where(Xi["redshift"]>s)
            pred[index] = lab
            results[i,j] = f1(yi,pred, lab)
    return results.mean(axis=0)

def choix_seuils(X,y,folds,nb_seuils): 
    """Fonction qui choisi les seuils en fonction des résultats de cross validation

    Args:
        X (pd.DataFrame): Jeu de données avec les co-variables sur lesquel on veut entrainer les valeurs seuils
        y (pd.DataFrame): Jeu de données avec la variable à prédire sur lequel on veut entrainer les valeurs seuils
        folds (int): nombre de folds (sous-divisions) pour faire la cross-validation
        nb_seuils (int): nombre de seuils à choisir

    Returns:
        seuil_0 (float): seuil choisi pour distinguer les classes 1 et 0
        seuil_2 (float): seuil choisi pour distinguer les classes 0 et 2
    """
    vect_x = np.linspace(X[["redshift"]].min()[0], X[["redshift"]].max()[0], nb_seuils)
    res_0 = frontiere(X,y, 0, folds, nb_seuils)
    res_2 = frontiere(X,y, 2, folds, nb_seuils)
    plt.plot(vect_x, res_0)
    plt.plot(vect_x, res_2)
    #On prend les seuils qui maximisent la f1 
    seuil_0 = vect_x[np.where(res_0==res_0.max())[0][0]]
    seuil_2 = vect_x[np.where(res_2==res_2.max())[0][0]]
    return seuil_0, seuil_2

def predict(val0,val2, X_te): 
    """Crée notre prédiction en fonction des valeurs val0 et val2 calculées précédamment. 
    Il crée un vecteur prédiction avec des valeurs: 
    - 1: redshift appartenant à ]-inf,val0[
    - 0: redshift appartenant à [val0,val2[
    - 2: redshift appartenant à [val2,+inf[

    Args:
        val0 (float): valeur seuil calculée pour distinguer la classe 1 et 0
        val2 (float): valeur seuil calculée pour distinguer la classe 0 et 2
        X_te (pd.DataFrame): vecteur des covariables test sur lesquels on va regarder la valeur de redshift

    Returns:
        pred: vecteur contenant les prédictions de notre méthode_seuil
    """
    pred = np.ones(X_te.shape[0])
    index_0 = np.where(X_te["redshift"]>val0)[0]
    index_2 = np.where(X_te["redshift"]>val2)[0]
    pred[index_0] = int(0)
    pred[index_2] = int(2)
    return pred.reshape(-1,1)

