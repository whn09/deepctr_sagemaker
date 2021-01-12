import os
import argparse

import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.utils import multi_gpu_model

from deepctr.models import *
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names

def main(model_dir, data_dir, train_steps, model_name):
    data = pd.read_csv(os.path.join(data_dir, 'criteo_sample.txt'))

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(),embedding_dim=4)
                           for i,feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
                          for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.2, random_state=2020)
    train_model_input = {name:train[name] for name in feature_names}
    test_model_input = {name:test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate
    if model_name == 'DeepFM':
        model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
    elif model_name == 'FNN':
        model = FNN(linear_feature_columns, dnn_feature_columns, task='binary')
    elif model_name == 'WDL':
        model = WDL(linear_feature_columns, dnn_feature_columns, task='binary')
    elif model_name == 'MLR':
        model = MLR(linear_feature_columns, dnn_feature_columns, task='binary')
    elif model_name == 'NFM':
        model = NFM(linear_feature_columns, dnn_feature_columns, task='binary')
    elif model_name == 'DIN':
        model = DIN(linear_feature_columns, dnn_feature_columns, task='binary')
    else:
        print(model_name+' is not supported now.')
        return
    
    gpus = int(os.getenv('SM_NUM_GPUS', '0'))
    print('gpus:', gpus)
    if gpus > 1:
        model = multi_gpu_model(model, gpus=gpus)
    
    model.compile("adam", "binary_crossentropy",
                  metrics=['binary_crossentropy'], )

    history = model.fit(train_model_input, train[target].values,
                        batch_size=256, epochs=train_steps, verbose=2, validation_split=0.2, )
    pred_ans = model.predict(test_model_input, batch_size=256)
    try:
        print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    except Exception as e:
        print(e)
    try:
        print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
    except Exception as e:
        print(e)
    
    model.save_weights(os.path.join(model_dir, 'DeepFM_w.h5'))

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    # For more information:
    # https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html
    args_parser.add_argument(
        '--data_dir',
        default='/opt/ml/input/data/training',
        type=str,
        help='The directory where the input data is stored. Default: /opt/ml/input/data/training. This '
             'directory corresponds to the SageMaker channel named \'training\', which was specified when creating '
             'our training job on SageMaker')

    # For more information:
    # https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html
    args_parser.add_argument(
        '--model_dir',
        default='/opt/ml/model',
        type=str,
        help='The directory where the model will be stored. Default: /opt/ml/model. This directory should contain all '
             'final model artifacts as Amazon SageMaker copies all data within this directory as a single object in '
             'compressed tar format.')

    args_parser.add_argument(
        '--train_steps',
        type=int,
        default=100,
        help='The number of steps to use for training.')
    
    args_parser.add_argument(
        '--model_name',
        default='DeepFM',
        type=str,
        help='Models: CCPM, FNN, PNN, WDL, DeepFM, MLR, NFM, AFM, DCN, DCNMix, DIN, DIEN, DSIN, xDeepFM, AutoInt, ONN, FGCNN, FiBiNET, FLEN.')
    
    args = args_parser.parse_args()
    main(**vars(args))
    