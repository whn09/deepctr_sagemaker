import os
import argparse

import json
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import *


def model_fn(model_dir):
    model = torch.load(os.path.join(model_dir, 'DeepFM.h5'))
    feat_index_dict = json.load(open(os.path.join(model_dir, "feat_index_dict.json"), 'r'))
    max_min = pd.read_csv(os.path.join(model_dir, 'max_min.txt'), sep='\t')
    return model, feat_index_dict, max_min


def input_fn(request_body, request_content_type):
#     print('[DEBUG] request_body:', type(request_body))
#     print('[DEBUG] request_content_type:', request_content_type)
    
    """An input_fn that loads a json"""
    if request_content_type == 'application/json':
        return json.loads(request_body)
    else:
        # Handle other content-types here or raise an Exception
        # if the content type is not supported.  
        return request_body

    
def predict_fn(input_data, model):
#     print('[DEBUG] input_data type:', type(input_data), input_data.shape)

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    feature_names = sparse_features+dense_features

    feat_index_dict = model[1]
    max_min = model[2]
#     print('feat_index_dict:', feat_index_dict)
#     print('max_min:', max_min)
    
    instances = input_data['instances']
    # print('before instances:', instances)
    for i in range(len(instances)):
        for feat in sparse_features:
            # print('feat:', feat)
            if instances[i][feat] is None:
                instances[i][feat] = '-1'
            instances[i][feat] = feat_index_dict[feat][instances[i][feat]]
        for feat in dense_features:
            max_min_feat = max_min[max_min['Unnamed: 0']==feat]
            # print('max_min_feat:', max_min_feat)
            if instances[i][feat] is not None:
                instances[i][feat] = (max_min_feat['max'].values[0]-instances[i][feat])/(max_min_feat['max'].values[0]-max_min_feat['min'].values[0])
    # print('after instances:', instances)
    
    instances_df = pd.DataFrame(instances)
    # instances_df[sparse_features] = instances_df[sparse_features].fillna('-1', )
    instances_df[dense_features] = instances_df[dense_features].fillna(0, )
    # print('instances_df:', instances_df)    
    # print('instances_df.info():', instances_df.info())
    
    test_model_input = {name: instances_df[name] for name in feature_names}
        
    pred = model[0].predict(test_model_input, 1)
#     print('[DEBUG] pred:', pred)
    
    result = pred
#     print('[DEBUG] result:', result)
    
    return result


# def output_fn(prediction, content_type):
#     pass


def get_integer_mapping(le):
    '''
    Return a dict mapping labels to their integer values from an SKlearn LabelEncoder
    le = a fitted SKlearn LabelEncoder
    '''
    res = {}
    for idx, val in enumerate(le.classes_):
        res.update({val:idx})
    return res


def main(model_dir, data_dir, train_steps, model_name, task, **kwargs):
    print(kwargs)
    data = pd.read_csv(os.path.join(data_dir, 'criteo_sample.txt'))
    
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    feat_index_dict = {}
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
        feat_index_dict.update({feat:get_integer_mapping(lbe)})
        
    # save category features index for serving stage
    with open(os.path.join(model_dir, "feat_index_dict.json"), 'w') as fo:
        json.dump(feat_index_dict, fo)

    # save min max value for each dense feature 
    s_max,s_min = data[dense_features].max(axis=0),data[dense_features].min(axis=0)
    pd.concat([s_max, s_min],keys=['max','min'],axis=1).to_csv(os.path.join(model_dir, 'max_min.txt'), sep='\t')    
    
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                              for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                              for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.2)

    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate

    device = 'cpu'
    use_cuda = True
    gpus = None
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'
        print('gpu count:', torch.cuda.device_count())
        gpus = []
        for i in range(torch.cuda.device_count()):
            gpus.append(i)
    print('gpus:', gpus)

    model = DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                   task='binary',
                   l2_reg_embedding=1e-5, device=device, gpus=gpus)

    model.compile("adagrad", "binary_crossentropy",
                  metrics=["binary_crossentropy", "auc"], )
    model.fit(train_model_input,train[target].values,batch_size=32,epochs=10,verbose=2,validation_split=0.0)

    pred_ans = model.predict(test_model_input, 256)
    print("")
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
    
    torch.save(model, os.path.join(model_dir, 'DeepFM.h5'))


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
        help='Models: CCPM, FNN, PNN, WDL, DeepFM, NFM, AFM, DCN, xDeepFM, AutoInt, FiBiNET.')
    
    args_parser.add_argument(
        '--task',
        default='binary',
        type=str,
        help='"binary" for binary logloss or "regression" for regression loss')
    
    # hyperparameters
    args_parser.add_argument(
        '--fm_group',
        default=['default_group'],
        type=list,
        help='group_name of features that will be used to do feature interactions.')
    args_parser.add_argument(
        '--dnn_hidden_units',
        default=(128, 128),
        type=list,
        help='list of positive integer or empty list, the layer number and units in each layer of DNN')
    args_parser.add_argument(
        '--cross_num',
        default=2,
        type=int,
        help='positive integet,cross layer number')
    args_parser.add_argument(
        '--cross_parameterization',
        default='vector',
        type=str,
        help='"vector" or "matrix", how to parameterize the cross network.')
    args_parser.add_argument(
        '--l2_reg_cross',
        default=1e-5,
        type=float,
        help='L2 regularizer strength applied to cross net')
    args_parser.add_argument(
        '--l2_reg_linear',
        default=1e-05,
        type=float,
        help='L2 regularizer strength applied to linear part')
    args_parser.add_argument(
        '--l2_reg_embedding',
        default=1e-05,
        type=float,
        help='L2 regularizer strength applied to embedding vector')
    args_parser.add_argument(
        '--l2_reg_dnn',
        default=0,
        type=float,
        help='L2 regularizer strength applied to DNN')
    args_parser.add_argument(
        '--seed',
        default=1024,
        type=int,
        help='to use as random seed.')
    args_parser.add_argument(
        '--dnn_dropout',
        default=0,
        type=float,
        help='float in [0,1), the probability we will drop out a given DNN coordinate.')
    args_parser.add_argument(
        '--dnn_activation',
        default='relu',
        type=str,
        help='Activation function to use in DNN')
    args_parser.add_argument(
        '--dnn_use_bn',
        default=False,
        type=bool,
        help='Whether use BatchNormalization before activation or not in DNN')
    args_parser.add_argument(
        '--low_rank',
        default=32,
        type=int,
        help='Positive integer, dimensionality of low-rank sapce.')
    args_parser.add_argument(
        '--num_experts',
        default=4,
        type=int,
        help='Positive integer, number of experts.')
    
    args = args_parser.parse_args()
    main(**vars(args))
    
#     null = None
#     fea_dict1 = {"I1":null,"I2":3,"I3":260.0,"I4":null,"I5":17668.0,"I6":null,"I7":null,"I8":33.0,"I9":null,"I10":null,"I11":null,"I12":0.0,"I13":null,"C1":"05db9164","C2":"08d6d899","C3":"9143c832","C4":"f56b7dd5","C5":"25c83c98","C6":"7e0ccccf","C7":"df5c2d18","C8":"0b153874","C9":"a73ee510","C10":"8f48ce11","C11":"a7b606c4","C12":"ae1bb660","C13":"eae197fd","C14":"b28479f6","C15":"bfef54b3","C16":"bad5ee18","C17":"e5ba7672","C18":"87c6f83c","C19":null,"C20":null,"C21":"0429f84b","C22":null,"C23":"3a171ecb","C24":"c0d61a5c","C25":null,"C26":null}
#     fea_dict2 = {"I1":null,"I2":-1,"I3":19.0,"I4":35.0,"I5":30251.0,"I6":247.0,"I7":1.0,"I8":35.0,"I9":160.0,"I10":null,"I11":1.0,"I12":null,"I13":35.0,"C1":"68fd1e64","C2":"04e09220","C3":"95e13fd4","C4":"a1e6a194","C5":"25c83c98","C6":"fe6b92e5","C7":"f819e175","C8":"062b5529","C9":"a73ee510","C10":"ab9456b4","C11":"6153cf57","C12":"8882c6cd","C13":"769a1844","C14":"b28479f6","C15":"69f825dd","C16":"23056e4f","C17":"d4bb7bd8","C18":"6fc84bfb","C19":null,"C20":null,"C21":"5155d8a3","C22":null,"C23":"be7c41b4","C24":"ded4aac9","C25":null,"C26":null}
#     fea_dict3 = {"I1":0.0,"I2":0,"I3":2.0,"I4":12.0,"I5":2013.0,"I6":164.0,"I7":6.0,"I8":35.0,"I9":523.0,"I10":0.0,"I11":3.0,"I12":null,"I13":18.0,"C1":"05db9164","C2":"38a947a1","C3":"3f55fb72","C4":"5de245c7","C5":"30903e74","C6":"7e0ccccf","C7":"b72ec13d","C8":"1f89b562","C9":"a73ee510","C10":"acce978c","C11":"3547565f","C12":"a5b0521a","C13":"12880350","C14":"b28479f6","C15":"c12fc269","C16":"95a8919c","C17":"e5ba7672","C18":"675c9258","C19":null,"C20":null,"C21":"2e01979f","C22":null,"C23":"bcdee96c","C24":"6d5d1302","C25":null,"C26":null}
#     fea_dict4 = {"I1":null,"I2":13,"I3":1.0,"I4":4.0,"I5":16836.0,"I6":200.0,"I7":5.0,"I8":4.0,"I9":29.0,"I10":null,"I11":2.0,"I12":null,"I13":4.0,"C1":"05db9164","C2":"8084ee93","C3":"02cf9876","C4":"c18be181","C5":"25c83c98","C6":null,"C7":"e14874c9","C8":"0b153874","C9":"7cc72ec2","C10":"2462946f","C11":"636405ac","C12":"8fe001f4","C13":"31b42deb","C14":"07d13a8f","C15":"422c8577","C16":"36103458","C17":"e5ba7672","C18":"52e44668","C19":null,"C20":null,"C21":"e587c466","C22":null,"C23":"32c7478e","C24":"3b183c5c","C25":null,"C26":null}

#     data = {"instances": [fea_dict1,fea_dict2,fea_dict3,fea_dict4]}
    
#     model = model_fn('/opt/ml/model')
#     input_data = input_fn(json.dumps(data), 'application/json')
#     result = predict_fn(input_data, model)
#     print('result:', result)