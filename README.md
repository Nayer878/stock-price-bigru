# Stock Price Prediction using BiGRU and Time Series Cross-Validation 

## 📄 Description 
Ce projet utilise un modèle Bidirectional GRU (BiGRU) pour prédire les prix de clôture d'une action (Apple - AAPL).
Nous utilisons une validation croisée temporelle pour une évaluation plus réaliste. 

## ⚙️ Technologies utilisées 
- Python 3.x 
- TensorFlow / Keras 
- Scikit-learn 
- Pandas 
- Matplotlib 

## 🧪 Méthodologie 
**Données** : Prix de clôture d'Apple (AAPL) de 2013 à 2022. 

**Prétraitement** : 
- Normalisation MinMaxScaler. 
- Création de séquences temporelles (window = 3). 

**Modèle** : 
- Deux couches BiGRU (64 et 256 unités). 
- Dropout pour régularisation. 
- Fonction de perte Huber. 

**Validation** : 
- Validation croisée temporelle (5 splits). 
- Early stopping pour éviter le surapprentissage. 

**Évaluation** : 
- MSE, MAE, R², SMAPE. 
- Directional Accuracy calculée sur le test set. 

## 📊 Résultats principaux 
- **R²** (test) : 0.9885 
- **MAPE** (test) : 1.98% 
- **Directional Accuracy** : 47.08% 

## 📁 Structure du projet
├── model_bigru.pdf        # Rapport complet
├── model_bigru.keras      # Modèle sauvegardé
├── scaler_minmax_gru.pkl  # Scaler sauvegardé
├── stock/AAPL.csv         # Données utilisées
├── README.md              # Ce fichier


## 🚀 Exécution rapide 
```bash
# Installer les dépendances 
pip install tensorflow scikit-learn pandas matplotlib joblib 

# Lancer le script principal 
python model_bigru.py 

## 📢 Objectif de cette publication
Je publie ce travail pour avoir des avis, des critiques et des suggestions d'amélioration avant de soumettre sur des plateformes de prépublication plus grandes.
N'hésitez pas à ouvrir une Issue pour toute remarque.

## 📬 Contact
Auteur : Mohamed Mouendhui Nael Email : mohamedmouendhuinael@gamil.com

Remarque :
Ce projet est un travail académique personnel, pas une recommandation financière.