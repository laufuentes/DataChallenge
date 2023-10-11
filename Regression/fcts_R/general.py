from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt 


def param_selection(param, mod, Xtr, ytr, Xte): 
    """_summary_

    Args:
        param (liste):  {"param1":['option1', 'option2'],"param2":[option1,option2,option3]}
        mod (sklearn.function): model(random_state=10)
        Xtr (pd.DataFrame): Co-variables du jeu de données train 
        ytr (pd.DataFrame): Variable à prédire du jeu de données train 
        Xte (pd.DataFrame): Co-variables du jeu de données test

    Returns:
        pred (np.array ou pd.DataFrame): vecteur contenant les prédictions du modèle selectionné par cross validation
    """

    grid_search = GridSearchCV(estimator=mod,param_grid=param,cv=5)
    grid_search.fit(Xtr, ytr)

    print("Best Hyperparameters: ", grid_search.best_params_)
    print("Best Score: ", grid_search.best_score_)

    pred = grid_search.predict(Xte)

    return pred 


def train_eval(model, X, y, X_test, y_test):
    """Fonction qui entraine un modèle, plotte et renvoit le r2_score sur le jeu de données test associé à la prédiction

    Args:
        model (sklearn.function): Modèle à entrainer pour faire la prediction
        X (pd.Dataframe): Co-variables du jeu de données train
        y (_type_): Variable à prédire du jeu de données train
        X_test (_type_): Co-variables du jeu de données test
        y_test (_type_): Variable à prédire du jeu de données test

    Returns:
        pred (np.array ou pd.DataFrame): prédiction sur le jeu de données test associé au modèle choisi en input
    """
    lab = str(model()) 
    #Entrainement du modèle 
    mod = model()
    mod.fit(X,y)
    pred = mod.predict(X_test)

    plt.figure()
    plt.hist(y_test, density=True, label="True", alpha=0.5, bins=np.linspace(3,9,7))
    plt.hist(pred, density=True, label=lab, alpha=0.5, bins=np.linspace(3,9,7))
    plt.title("Histogramme des predicitions de "+lab)
    plt.legend()
    plt.show()

    print("normal: ", r2_score(y_test, pred))
    return pred

def kernel_regression_decision(X, Y, lbda=1, gamma=1):
    """Fonction qui calcule la regression ridge avec noyau gaussien

    Args:
        X (np.array): Co-variables
        Y (np.array): Variable à prédire
        lbda (int, optional): Defaults to 1.
        gamma (int, optional): _description_. Defaults to 1.

    Returns:
        np.sum(alpha*K, axis=1) (np.array): The prediction for X
    """
    # Gaussian Kernel matrix
    K = np.exp(-gamma*(X[:,np.newaxis]-X[:, np.newaxis].T)**2)
    # Solution of Kernel Ridge Regression
    alpha = np.linalg.inv(K + len(X)*lbda*np.eye(len(X)))@Y
    # Return the predictions for X
    return np.sum(alpha*K, axis=1)


def build_pred(X_test0, X_test1, pred0,pred1): 
    """Fonction qui combine les prédictions de data0 et data1 

    Args:
        X_test0 (pd.DataFrame): Vecteur contenant les co-variables à wine_type=0 du jeu de données test 
        X_test1 (pd.DataFrame): Vecteur contenant les co-variables à wine_type=0 du jeu de données test 
        pred0 (np.array ou pd.DataFrame): prédictions associés pour les indices de wine_type=0
        pred1 (np.array ou pd.DataFrame): prédictions associés pour les indices de wine_type=1

    Returns:
        pred (np.array): Vecteur contenant l'ensemble des prédictions du jeu de données test 
    """
    data_0 = np.column_stack((X_test0[["wine_ID"]].to_numpy(), pred0))
    data_1 = np.column_stack((X_test1[["wine_ID"]].to_numpy(), pred1))
    pred = np.row_stack((data_0,data_1))
    return pred 

