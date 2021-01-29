from pathlib import Path
import requests
import pickle
import gzip


DATA_PATH = Path('data')
PATH = DATA_PATH / 'mnist'
PATH.mkdir(parents=True, exist_ok=True)

URL = 'https://github.com/pytorch/tutorials/raw/master/_static/'
FILE_NAME = 'mnist.pkl.gz'

if not (PATH / FILE_NAME).exists():
    content = requests.get(URL + FILE_NAME).content
    (PATH / FILE_NAME).open('wb').write(content)