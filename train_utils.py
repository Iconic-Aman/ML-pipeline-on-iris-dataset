import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import (
    LinearRegression, LogisticRegression,
    Ridge, Lasso, ElasticNet, SGDClassifier, SGDRegressor
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import GridSearchCV
import json

#load data from data path
with open('jsonFile.json', 'r') as json_file:
    loadData = json.load(json_file)
    
data = loadData['design_state_data']

#  extract imp info
algo = data['algorithms']
target = data['target']['target']
task_type = data['target']['type']
dataset = data['session_info']['dataset']
columns = list(data['feature_handling'].keys())

#feature selection
selected_features = []
for feature, val in data['feature_handling'].items():
    if val['is_selected']:
        strategy = val['feature_details'].get('impute_with')
        impute_val = val['feature_details'].get('impute_value')

        selected_features.append(feature)

#feature reduction
feature_reduction_method  = data['feature_reduction']['feature_reduction_method']
num_of_trees = data['feature_reduction']['num_of_trees']
depth_of_trees = data['feature_reduction']['depth_of_trees']



#model registry
model_registry = {
    'RandomForestClassifier' : RandomForestClassifier,
    'RandomForestRegressor': RandomForestRegressor,
    # 'GBTClassifier' : GBTClassifier,
    # 'GBTRegressor': GBTRegressor,
    'LinearRegression': LinearRegression,
    'LogisticRegression': LogisticRegression,
    'RidgeRegression': Ridge,
    'LassoRegression': Lasso,
    'ElasticNetRegression': ElasticNet,
    # 'xg_boost': XGBClassifier,
    'DecisionTreeRegressor': DecisionTreeRegressor,
    'DecisionTreeClassifier': DecisionTreeClassifier,
    'SVM': SVC,
    'SGD': SGDClassifier,
    'KNN': KNeighborsClassifier,
    "MLPClassifier": MLPClassifier,
    "MLPRegressor": MLPRegressor
}

def get_param_grid(model_name, config):
    if model_name == 'RandomForestClassifier' or model_name == 'RandomForestRegressor':
        return {
                'n_estimators': [config['min_trees'], config['max_trees']],
                'max_depth': [config['min_depth'], config['max_depth']],
                'min_samples_leaf': [config['min_samples_per_leaf_min_value'], config['min_samples_per_leaf_max_value']]
        }
    
    elif model_name == 'RidgeRegression':
        return{
                'alpha': [config['min_regparam'], config['max_regparam']],
                'max_iter': [config['min_iter'], config['max_iter']]
            
        }
    elif model_name == 'LassoRegression':
        return{
                'alpha': [config['min_regparam'], config['max_regparam']],
                'max_iter': [config['min_iter'], config['max_iter']]
            
        }
    
    elif model_name == "ElasticNetRegression":
        return{
                'alpha': [config['min_regparam'], config['max_regparam']],
                'max_iter': [config['min_iter'], config['max_iter']],
                'l1_ratio': [config['min_elasticnet'], config['max_elasticnet']]
            
        }
    elif model_name == "DecisionTreeRegressor" or model_name == 'DecisionTreeClassifier':
        return{
                'max_depth': [config['min_depth'], config['max_depth']],
                'min_samples_leaf': [config['min_samples_per_leaf']]
            
        }
    elif model_name == "SVM":
        return{
                'max_iter': [config['max_iterations']],
                'C': config['c_value'],
                'kernel': [
                    k for k, use in {
                        'linear': config.get('linear_kernel', False),
                        'rbf': config.get('rep_kernel', False),   # assuming rep = rbf
                        'poly': config.get('polynomial_kernel', False),
                        'sigmoid': config.get('sigmoid_kernel', False)
                    }.items() if use
                ]
            
        }
    elif model_name == "KNN":
        return{
                'n_neighbors': config['k_value'],
                'weights': ['distance'] if config.get('distance_weighting') else ['uniform'], 
                'p': config.get('p_value', 2)
            
        }
    elif model_name == "extra_random_trees":
        return{
                "n_estimators": config["num_of_trees"],
                "max_depth": config["max_depth"],
                "min_samples_leaf": config["min_samples_per_leaf"]
            
        }
    elif model_name == "ElasticNetRegression":
        return{
                'hidden_layer_sizes' : [(67,), (89,)], #must be tuple
                'activation': ['relu', 'tanh'] if not config['activation'] else config['activation'],
                'alpha': [0.002] if config['alpha_value'] == 0 else [config["alpha_value"]],
                'max_iter': [200] if config['max_iterations'] == 0 else [config['max_iterations']],
                "early_stopping": [config.get("early_stopping", True)],
                'solver': [config.get('solver').lower()],
                'shuffle': [config.get('shuffle_data', True)]
            
        }
    else:
        return None
    
# Hyperparameter Tuning and train model
def train_selected_model(X_train, X_test, y_train, y_test, algo, model_registry, get_param_grid, hyperparameters):
    for model_name, config in algo.items():
        if config.get('is_selected'):
            print(f'Here, we are using {model_name} model.')
            
            #get the model class from model registry
            model_class = model_registry.get(model_name)
            if model_class is None:
                print(f'{model_name} is not registered in model_registry')
                continue
            
            model = model_class()
            param_grid = get_param_grid(model_name, config)
            if param_grid is None:
                print(f'no param grid found for this {model_name}')
                continue
            
            #Apply grid search cv and train the model
            grid_search = GridSearchCV(
                estimator= model,
                param_grid= param_grid,
                cv = data['hyperparameters']['num_of_folds'],
                n_jobs= data['hyperparameters']['parallelism'],
                verbose= 3,
            )
            
            grid_search.fit(X_train, y_train)
            
            print(f" Best Params for {model_name}: {grid_search.best_params_}")
            print(f" Test Score: {grid_search.score(X_test, y_test)}\n")



if __name__ == '__main__':
    print(f'feature_reduction_method : {feature_reduction_method}')
    print(f'number of trees : {num_of_trees}')
    print(f'depth of tress : {depth_of_trees}')
    
