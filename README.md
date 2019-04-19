# Real-time parking lot occupancy detection using Deep Learning

This repo has source code and model files for the modified implementation of [Real-time image-based parking occupancy detection using deep learning, Acharya, D., Yan, W., &amp; Khoshelham, K. 2018](http://ceur-ws.org/Vol-2087/paper5.pdf) work.

The authors implemented the model using pre-trained VGG network and Support Vector Machines (SVM). VGG network is used for feature extractions and SVM is used for classificaition. They used PKLot dataset for training and evaluated on custom dataset created by authors.

I exprimented with ResNet50 and VGG16 for featre extraction and used SVM and CNN for classification. SVM model with ResNet-50 gave better performance of 99% of average f1-score compared to all other models. The models are trained and evaluated on PKLot dataset. Please find the model metrics in the respective notebooks

## Notebooks
Following are the notebooks you can find in the repo,
- create_dataset_index – This notebook is used to created dataset for all the models
- cnn_models_resnet50 – Notebook for RestNet-50 model training
- cnn_models_vgg16 – Notebook for VGG-16 model training
- cnn_models_from_scratch – CNN model training from scratch
- resnet50_linear_svm_model – SVM with ResNet-50 model training


## Model framework
![Model framework](images/model_framework.JPG)

## Dataset
![Dataset](images/pklot_dataset.JPG)

PKLot dataset - https://web.inf.ufpr.br/vri/databases/parking-lot-database/

