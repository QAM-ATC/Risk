from setuptools import setup, find_packages

with open('README.md') as readme_file:
    README = readme_file.read()

with open('HISTORY.md') as history_file:
    HISTORY = history_file.read()

setup_args = dict(
    name='quant_risk',
    version='1.1.0',
    description='Quantitative functions in Python',
    long_description_content_type="text/markdown",
    long_description=README + '\n\n' + HISTORY,
    license='MIT',
    packages=find_packages(),
    author=' NTU Quantitative Asset Management Club',
    author_email='quantassetmgmtdivision@gmail.com',
    keywords=['Quantitative', 'Risk', 'Portfolio'],
    url='https://github.com/QAM-ATC/Risk/tree/package',
)

install_requires = [
    'statsmodels==0.12.2',
    'empyrical==0.5.5',
    'matplotlib==3.4.2',
    'pyportfolioopt==1.4.2',
    'sklearn',
    'pandas',
    'numpy'
]

if __name__ == '__main__':
    setup(**setup_args, install_requires=install_requires)