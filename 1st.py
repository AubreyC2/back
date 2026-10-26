import torch
import requests
import zipfile
from pathlib import Path

#setup path to a data_name folder
data_path= Path("../data/")
data_set= data_path / "fixed_ideal"

#if dne download, if e then dont download

if data_set.is_dir():
    print(f'{data_set} dir exist .. skip download')
else:
    print(f'{data_set} dne, creating one.')
    data_set.mkdir(parents=True, exist_ok= True)

#lets download
with open("../unknown/data_set", "wb") as f:
    request = requests.get("https://raw.githubusercontent.com/HungryForPi/AI-Experiments/refs/heads/main/GroebnerTime/FixedIdeal/data.txt")
    print("Downloading data_name set..")
    f.write(request.content)

