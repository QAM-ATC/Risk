[![Documentation Status](https://readthedocs.org/projects/quant-risk/badge/?version=latest)](https://quant-risk.readthedocs.io/en/latest/?badge=latest) ![GitHub](https://img.shields.io/github/license/QAM-ATC/Risk)
# Risk
Welcome to the Quant-Risk package!

To set up your environment:
1. install pipenv by:
``` {python}
pip install pipenv
```
2. then go to the main directory of this repo and do:
``` {python}
pipenv shell
```
3. Update your setup tools package by:
```{python}
python -m pip install -U pip setuptools
```
3. Now, to install the package, run the command:
``` {python}
pipenv install quant_risk
```
4. To exit the virtual environment do:
``` {python}
exit
```

Since, PyPortfolioOpt requires  Visual Studio C++, do the following:
1. Before installing and setting up your environment, install Visual Studio Build Tools from here: https://visualstudio.microsoft.com/downloads/
choose the Community version if you have Windows
2. From the available softwares, select "Visual Studio Build Tools 2019"
3. Now your pipenv shell should be able to install PyPortfolioOpt

Note: The above is for Windows users, for Mac users please see: https://osxdaily.com/2014/02/12/install-command-line-tools-mac-os-x/

To run a jupyter notebook instance with this venv:
1. Activate your venv by:
``` {python}
pipenv shell
```
2. run the command:
``` {python}
pipenv install jupyter
```
3. run the command:
``` {python}
pipenv run jupyter notebook
```
All done!
An instance of Jupyter notebook should now open up in your browser
