# Real-time parking lot occupancy detection using Deep Learning

This repo has source code and model files for the modified implementation of [Real-time image-based parking occupancy detection using deep learning, Acharya, D., Yan, W., &amp; Khoshelham, K. 2018](http://ceur-ws.org/Vol-2087/paper5.pdf) work.

The authors implemented the model using pre-trained VGG network and Support Vector Machines (SVM). VGG network is used for feature extractions and SVM is used for classificaition. They used PKLot dataset for training and evaluated on custom dataset created by authors.

I exprimented with ResNet50 and VGG16 for featre extraction and used SVM and CNN for classification. SVM model with ResNet-50 gave better performance of average f1-score compared to all other models. The models are trained and evaluated on PKLot dataset. Please find the model metrics in the respective notebooks

## Notebooks
Following are the notebooks you can find in the repo,
- Dataset creation - [create_dataset_index.ipynb](create_dataset_index.ipynb)
- ResNet50 - [cnn_models_resnet50.ipynb](cnn_models_resnet50.ipynb)
- VGG16 - [cnn_models_vgg16.ipynb](cnn_models_vgg16.ipynb)
- CNN from scratch - [cnn_models_from_scratch.ipynb](cnn_models_from_scratch.ipynb)
- ResNet50 + SVM - [resnet50_linear_svm_model.ipynb](resnet50_linear_svm_model.ipynb)


## Model framework
![Model framework](images/model_framework.JPG)

## Dataset
![Dataset](images/pklot_dataset.JPG)

PKLot dataset - https://web.inf.ufpr.br/vri/databases/parking-lot-database/

