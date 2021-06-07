# Risk
This is going to be the Risk codebase.

To set up your environment:
1. install pipenv by: pip install pipenv
2. then go to the main directory of this repo and do: pipenv shell
3. Ideally, pipenv should detect the Pipfile or the requirements.txt, if not then:
    manually run the command: pipenv install -r requirements.txt
4. To exit the virtual environment do: exit

Note: please let me know if you need any package as if one person updates their venv without others doing the same, it'll mess up the venv.

To run a jupyter notebook instance with this venv:
1. Activate your venv by: pipenv shell
2. run the command: pipenv run jupyter notebook
All done!
An instance of Jupyter notebook should now open up in your browser