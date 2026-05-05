# Projet d'Analyse Numerique

Ce projet implemente des methodes numeriques d'interpolation et d'integration pour analyser des donnees experimentales.

## Contenu du projet

- Interpolation polynomiale (Lagrange et Newton)
- Integration numerique (rectangle, trapezes, Simpson, adaptative, Gauss-Legendre)
- Application au refroidissement d'un composant
- Application a l'ecoulement dans un canal
- Generation de graphiques et sauvegarde automatique des resultats

## Structure

- core/
	- polynomial_interpolation.py
	- numeric_integration.py
- models/
	- temperatures.py
	- flow.py
- utils/
	- visualizer.py
- tests/
	- test_polynomial_interpolation.py
	- test_numeric_integration.py
	- test_applications.py
- main.py

## Prerequis

- Python 3.11+
- pip

## Installation

1. Creer un environnement virtuel (optionnel mais recommande):

```powershell
python -m venv .venv
```

2. Activer l'environnement:

```powershell
.venv\Scripts\Activate.ps1
```

3. Installer les dependances:

```powershell
pip install -r requirements.txt
```

## Lancer le programme principal

Execution standard:

```powershell
python main.py
```

Execution sans affichage de fenetres (utile en environnement headless):

```powershell
python main.py --no-show
```

Les sorties sont sauvegardees dans le dossier `results/` (cree automatiquement):

- phenomene_runge.png
- comparaison_interpolations_refroidissement.png
- analyse_refroidissement.png
- analyse_ecoulement.png
- convergence_integration.png
- summary.json

## Lancer les tests

Le projet utilise `unittest` (bibliotheque standard):

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

## Notes

- Si le style Matplotlib demande n'est pas disponible, une valeur par defaut est appliquee automatiquement.
- Les methodes d'integration et d'interpolation incluent des validations d'entrees pour eviter les erreurs silencieuses.
