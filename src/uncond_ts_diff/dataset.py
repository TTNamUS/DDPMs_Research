# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import tarfile
from pathlib import Path
from urllib import request
import datetime
import requests
import bs4
import yfinance as yf
from gluonts.dataset.common import ListDataset,MetaData,TrainDatasets
from gluonts.dataset.common import load_datasets
from gluonts.dataset.repository.datasets import get_dataset, get_download_path
from gluonts.dataset.split import split
import numpy as np


default_dataset_path: Path = get_download_path() / "datasets"
wiki2k_download_link: str = "https://github.com/awslabs/gluonts/raw/b89f203595183340651411a41eeb0ee60570a4d9/datasets/wiki2000_nips.tar.gz"  # noqa: E501


def get_gts_dataset(dataset_name):
    if dataset_name == "wiki2000_nips":
        wiki_dataset_path = default_dataset_path / dataset_name
        Path(default_dataset_path).mkdir(parents=True, exist_ok=True)
        if not wiki_dataset_path.exists():
            tar_file_path = wiki_dataset_path.parent / f"{dataset_name}.tar.gz"
            request.urlretrieve(
                wiki2k_download_link,
                tar_file_path,
            )

            with tarfile.open(tar_file_path) as tar:
                tar.extractall(path=wiki_dataset_path.parent)

            os.remove(tar_file_path)
        return load_datasets(
            metadata=wiki_dataset_path / "metadata",
            train=wiki_dataset_path / "train",
            test=wiki_dataset_path / "test",
        )
    else:
        return get_dataset(dataset_name)

def get_crypto( prediction_length: int,freq: str="H",split_offset: int = None):
    start_date = '2023-01-01'
    #change format date
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    print(start_date)
    list_crypto = []
    url='https://coinmarketcap.com/coins/'
    res = requests.get(url)
    soup = bs4.BeautifulSoup(res.content, 'html.parser')
    # get data from class "sc-4984dd93-0 iqdbQL coin-item-symbol"
    data = soup.find_all("p", class_="sc-4984dd93-0 iqdbQL coin-item-symbol")
    for i in data:
        if 'USD' not in i.text:
            print(i.text)
            list_crypto.append(i.text)

    list_data=[]
    for crypto in list_crypto:    
        data = yf.download(crypto+'-USD', interval='1h', start=start_date,end="2024-01-01")
        data.reset_index()
        data = data.rename(columns={ 'Close':'target'})
        list_data.append(data['target'].values.tolist())


    list_data=list_data[:5]
    training_data=[]
    test_data=[]    
    for price in list_data:
            #split data to train and test
            split = int(len(price)-24)
            train = price[:split]
            test = price
            print(len(train),len(test))
            i=list_data.index(price)
            training_data.append({"start": start_date, "target": train,"feat_static_cat":np.array([i])})
            test_data.append({"start": start_date, "target": test,"feat_static_cat":np.array([i])})
            


    dataset_train = ListDataset(
        data_iter=training_data,
        freq=freq
    )    
    dataset_test = ListDataset(
        data_iter=test_data,
        freq=freq
    )
    #merge
    dataset = TrainDatasets(
        metadata=MetaData(
            freq=freq,
            prediction_length=prediction_length,
            start=start_date,
        ),
        train=dataset_train,
        test=dataset_test,
    )

    return dataset

if __name__ == "__main__":
    data= get_crypto(prediction_length=24)
    print(data)
    
