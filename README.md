# Méthodes d'apprentissage supervisé: Data Challenge 
Benali Nafissa, Fuentes Vicente Laura 

M2 Mathématiques et Intelligence Artificielle 2023/2024 

## Introduction: 
Bienvenu sur notre repositoire GitHub pour Méthodes d'apprentissage supervisé et Data Challenge. Dans le cadre du Data Challenge nous avons été amenés à construire des modèles prédictifs pour deux problèmes différents, un de classification et un de régréssion: 
- Classification: "Classification d'objets célestes"
- Régréssion: "Prédiction de la qualité d'un vin"

Pour organiser ce repositoire, nous avons décidé de creer deux dossiers (Regression et Classification). Dans chacun de ces dossiers, vous pourrez trouver les notebooks python montrant notre démarche amenant au choix du modèle final.  

## Guide du repositoire:
**0- Etape preliminaire: Création d'un environnement conda:**
Pour éviter des problèmes, nous proposons de créer un environnement conda adapté à nos besoins. Celui-ci servirá ultérieurement comme kernel pour les notebooks. Pour cela, il suffit d'executer dans le terminal (au niveau de notre repositoire), les suivantes commandes: 
```
conda create --name <env_name> --file env_requirements.txt
```

**1- Localisation des datasets train et test**
Pour pouvoir implémenter les notebooks, il faudra bien placer chaque dataset dans sa localisation correspondante: 
- Classification: 
> /Classification/Dataset_C
- Régréssion: 
> /Regression/Dataset_R

**2- Implémentation des notebooks**
Pour chacun des problèmes, on pourra suivre les étapes de notre démarche en implémentant les notebooks de chaque dossier dans l'ordre. 

*NB: Pour pouvoir lancer les notebooks, il faudra choisir l'environnement conda crée précédamment en tant que kernel.*

