# Risk
This is going to be the Risk codebase.

To set up your environment:
1. install pipenv by:
``` {python}
pip install pipenv
```
2. then go to the main directory of this repo and do:
``` {python}
pipenv shell
```
3. Ideally, pipenv should detect the Pipfile or the requirements.txt, if not then:
    manually run the command:
``` {python}
pipenv install -r requirements.txt
```
4. To exit the virtual environment do:
``` {python}
exit
```
Note: please let me know if you need any package as if one person updates their venv without others doing the same, it'll mess up the venv.
    : while installing packages, please change 'requirements.txt' by 'requirementsMac.txt' or 'requirementsWindows.txt' depending upon which OS you have

Since, PyPortfolioOpt requires very specific versions of numpy and pandas along with Visual Studio C++, do the following:
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
pipenv run jupyter notebook
```
All done!
An instance of Jupyter notebook should now open up in your browser
