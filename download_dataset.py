import kaggle

kaggle.api.authenticate()

kaggle.api.dataset_download_files('huangruichu/multiperson', path='dataset', unzip=True)





# import opendatasets as od

# od.download("https://www.kaggle.com/datasets/huangruichu/multiperson/data","dataset")