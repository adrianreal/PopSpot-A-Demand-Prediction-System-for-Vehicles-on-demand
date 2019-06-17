import numpy as np
import sys
import pandas as pd
import lightgbm as lgb
import argparse
import pickle
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

### Function to calculate root mean square error.
def rmse(y_true, y_pred): 
        return np.sqrt(np.mean((y_true - y_pred)**2))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_data', type=str)
	parser.add_argument('--model', type=str)
	args = parser.parse_args()
    	file_path = args.train_data
	model = args.model
	train_data = pd.read_csv(file_path, skiprows = 0)
	no_tuples,no_features = train_data.shape

	# Separate the target variable column from the feature columns.
	y_train = train_data['0']
	x_train = train_data.drop(['0'], axis=1)

	# Depending on the user's choice of model, conduct training using y_train and x_train.
	if model == 'lgbm':
		# If model chosen is LightGBM.
		print "Training Begins"
		lgb_train = lgb.Dataset(x_train,y_train)
		params = {'task': 'train', 'boosting_type': 'gbdt', 'objective':'regression', 'metric':('mape','l2_root','l1'), 'metric_freq':1, 'is_training_metric':'true','max_bin':255,'num_trees':4000,'learning_rate':0.005,'num_leaves':31,'tree_learner':'serial','feature_fraction':0.9,'bagging_freq':5,'min_data_in_leaf':1000,'min_sum_hessian_in_leaf':5.0,'is_enable_sparse':'true','use_two_round_loading':'false','num_threads':28}
		evals_result = {}

		gbm = lgb.train(params, lgb_train, num_boost_round=4000, valid_sets=[lgb_train], categorical_feature=[0,1,2,3,4,5],evals_result=evals_result,verbose_eval=10)
		print('Feature names: ', gbm.feature_name())
		print ('Feature importances: ', list(gbm.feature_importance(importance_type='gain')))
		print 'Saving Model'
		gbm.save_model('models/lgbm.txt')
		print "Training Completed"

	else:
			# LightGBM API automatically embeds categorical variables. If the mode chosen is not LightGBM, one-hot encoding is done here to embed the categorical features.
	        enc = OneHotEncoder(n_values = "auto", categorical_features = [0,1,2,3,4,5])
	        x_train = enc.fit_transform(x_train).toarray()
		pickle.dump(enc,open("models/one_hot","wb"))

		if model == 'ridge':
			# If model chosen is Ridge Regression.
			print "Training Begins"
			reg = Ridge(pow(10,-4), normalize = False)
			reg.fit(x_train, y_train.ravel())
			pickle.dump(reg,open("models/ridge","wb"))
			print "Training Completed"

		if model == 'linear':
			# If model chosen is Linear Regression.
			print "Training Begins"
			reg = LinearRegression()
			reg.fit(x_train, y_train.ravel())
			pickle.dump(reg,open("models/linear","wb"))
			print "Training Completed"

		elif model == 'mlp':
			# If model chosen is Multi-layer Perceptron.
			print "Training Begins"
			reg = MLPRegressor(hidden_layer_sizes=(10,),  activation='relu', solver='adam',    alpha=0.001,batch_size='auto',
               learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
               random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
               nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
               epsilon=1e-08)

			reg.fit(x_train, y_train.ravel())
			pickle.dump(reg,open("models/mlp","wb"))
			print "Training Completed"
		else:
			print "Please input a model from {lgbm, ridge, mlp}"
