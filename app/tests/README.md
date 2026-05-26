# Tests du pipeline ML

Ce dossier ne fait pas partie de l'execution normale de l'application.
Il sert a valider manuellement, avec `pytest`, des proprietes critiques du
pipeline de prevision MASI.

## Lancer les tests

Depuis le dossier `app/` :

```powershell
..\.venv\Scripts\python.exe -m pytest tests
```

Lancer un seul test :

```powershell
..\.venv\Scripts\python.exe -m pytest tests\test_no_leakage.py
..\.venv\Scripts\python.exe -m pytest tests\test_sequences.py
```

## test_no_leakage.py

Ce test verifie la logique anti-leakage du split temporel.

Il charge `app/data/final/master_dataset.csv`, construit le dataset modele,
puis applique `chronological_train_test_split`.

Il verifie que :

- le bloc d'apprentissage complet contient `train_window` observations ;
- le bloc de test contient la fenetre derivee de `test_window_ratio` ;
- la derniere date du bloc train est strictement avant la premiere date du bloc test.

Son objectif est de confirmer que le modele ne s'entraine pas sur des donnees
futures.

## test_sequences.py

Ce test verifie la construction des sequences LSTM.

Il charge le dataset final, construit le split chronologique, applique le scaler
sur le bloc train, puis appelle `make_sequences_from_block`.

Il verifie que :

- chaque sequence contient `seq_len` observations passees ;
- le nombre de features dans chaque sequence est correct ;
- les tableaux `X`, `y` et `dates` sont alignes ;
- la premiere date cible correspond a la fin de la premiere fenetre temporelle.

Son objectif est de confirmer que le LSTM recoit une fenetre passee coherente,
sans information future.

## Formulation entretien

Le dossier `tests` n'est pas utilise par le dashboard ni par FastAPI pendant
l'execution normale. Il est utilise avec `pytest` pour valider deux garanties
methodologiques du pipeline ML : l'absence de leakage temporel et la construction
correcte des sequences LSTM.
