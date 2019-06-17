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

### Functions to calculate root-mean-square error, mean absolute error, and mean absolute percentage error respectively.
def rmse(y_true, y_pred): 
        return np.sqrt(np.mean((y_true - y_pred)**2))

def mae(y_true, y_pred): 
        return np.mean(abs(y_true - y_pred))

def mape(y_true, y_pred):
	idxList = np.where(y_true == 0)
	y_t = np.delete(y_true,idxList)
	y_p = np.delete(y_pred,idxList)
        return np.mean(np.abs((y_t - y_p)/y_t))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--test_data', type=str)
	parser.add_argument('--model', type=str)
	args = parser.parse_args()

        file_path = args.test_data
	test_data = pd.read_csv(file_path, skiprows = 0)
	no_tuples,no_features = test_data.shape

	# Separate the target variable column from the feature columns
	y_test = test_data['0'].values
	x_test = test_data.drop(['0'], axis=1).values
	
	lgb_test = lgb.Dataset(x_test,y_test)

	# Depending on the user's choice of model, conduct testing using y_test and x_test
	if args.model == 'lgbm':
		# If model chosen is LightGBM
		print 'Testing Begins'
		gbm = lgb.Booster(model_file='models/lgbm.txt')
		pred_results = gbm.predict(x_test)
        	root_mse = rmse(y_test.ravel(),pred_results)
		mean_ae = mae(y_test.ravel(),pred_results)
		print 'RMSE: '+str(root_mse)
		print 'MAE: '+str(mean_ae)
		print 'Testing Completed'

	else:
		# LightGBM API automatically embeds categorical variables. If the mode chosen is not LightGBM, one-hot encoding is done here to embed the categorical features.
		enc = pickle.load(open('models/one_hot','rb'))
	        x_test = enc.transform(x_test).toarray()

		if args.model == 'ridge':
			# If model chosen is Ridge Regression.
			print 'Testing Begins'
			reg = pickle.load(open('models/ridge','rb'))
			pred_results = reg.predict(x_test)
	        	root_mse = rmse(y_test.ravel(),pred_results)
			mean_ae = mae(y_test.ravel(),pred_results)
			print 'RMSE: '+str(root_mse)
			print 'MAE: '+str(mean_ae)
			print 'Testing Completed'

		elif args.model == 'linear':
			# If model chosen is Linear Regression.
			print 'Testing Begins'
			reg = pickle.load(open('models/linear','rb'))
			pred_results = reg.predict(x_test)
	        	root_mse = rmse(y_test.ravel(),pred_results)
			mean_ae = mae(y_test.ravel(),pred_results)
			print 'RMSE: '+str(root_mse)
			print 'MAE: '+str(mean_ae)
			print 'Testing Completed'


		elif args.model == 'mlp':
			# If model chosen is Multi-layer Perceptron.
			print 'Testing Begins'
			reg = pickle.load(open('models/mlp','rb'))
			pred_results = reg.predict(x_test)
	        	root_mse = rmse(y_test.ravel(),pred_results)
			mean_ae = mae(y_test.ravel(),pred_results)
			print 'RMSE: '+str(root_mse)
			print 'MAE: '+str(mean_ae)
			print 'Testing Completed'
		else:
                        print 'Please input a model from {lgbm, ridge, mlp}'	
