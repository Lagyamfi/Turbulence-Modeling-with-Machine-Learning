"""
Collection of classes and functions that are used for making tests
"""

#import libraries
import matplotlib.pyplot as plt
plt.style.use('seaborn-pastel')
import numpy as np
import pandas as pd
import clean_dataframe as cd
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms


# sklearn libraries
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

# xgboost libraries
import xgboost as xgb

class Make_test(object):
    """
    Class for making test using a saved model.
    """
    def __init__(self, model, test_file):
        self.model = model
        self.test_file = cd.clean_dataframe(test_file)
        self.x_actual, self.y_actual = cd.X_Y_split(self.test_file)
        self.y_actual = self.y_actual.values.ravel()
        self.predictions = self.make_predictions()
        self.test_file["predictions"] = self.predictions

    def __call__(self):
        return self.test_file

    def __str__(self):
        pass

    def __repr(self):
        pass
     
    def make_predictions(self):
        self.predictions = self.model.predict(self.x_actual)
        return self.predictions
    
    def get_full_dataframe(self):
        self.full_dataframe = self.test_file.copy()
        self.full_dataframe["predictions"] = self.predictions
        return self.full_dataframe

    def compare_values(self,return_values=False, threshold=0, plot=False):
        """
        Compare predicted values with actual response values. 
        Specify threshold to get the rows with the maximum difference between
        both values equal or greater than threshold
        Specify plot to get plot of comparison
        """
        comparison = pd.DataFrame(dict(ACTUALS=self.y_actual, PREDICTIONS=self.predictions))
        if threshold:
            comparison = comparison[np.abs(comparison.ACTUALS - comparison.PREDICTIONS >= threshold)]
        if plot:
            plot_predictions(comparison, colormap=True)
        if return_values:
            return comparison

    def score(self, metric="rmse", plot_residuals=False):
        """
        Get score of model with test data depending on which metric.
        RMSE is the default.
        Specify plot_residuals to show the distribution of the residuals and
        the fitted values
        """
        error = self.y_actual - self.predictions
        error_squared = np.square(error)
        mae = np.mean(error)
        mse = np.mean(error_squared)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.predictions, self.y_actual)
        if metric == "rmse":
            self.test_score = rmse
        elif metric == "mse":
            self.test_score = mse
        elif metric == "mae":
            self.test_score = mae
        elif metric == "r2":
            self.test_score = r2
        if plot_residuals:
            fig, ax = plt.subplots()
            ax.scatter(self.predictions, self.predictions-self.y_actual,
                        c="b", s=40, alpha=0.5)
            ax.axhline(lw=2, color="red")
            ax.set(title="Residuals of errors on Test data")
            ax.set_ylabel("Residuals")
            ax.set_xlabel("Predicted Values")
        return float(self.test_score)
    
    def make_plot(self, title="Test", save=False):
        self.full_dataframe = self.test_file
        grid = int(np.cbrt(len(self.test_file)) // 2)
        plot_wall_test(self.full_dataframe, title ,half_channel_grids=grid, prediction=True, save=False)

#********* CLASS FOR ONLY XGBOOST TESTING ************************************
class XGB_test(Make_test):
    """
    Create an xgb_boost test object. inherited from main testing class (Make
    test)
    """
    import xgboost as xgb
    def __init__(self, model, test_file):
        Make_test.__init__(self, model, test_file)


    def make_predictions(self):
        self.test_matrix = xgb.DMatrix(data=self.x_actual, feature_names=self.x_actual.columns)
        try:
            self.predictions = self.model.predict(self.test_matrix)
        except TypeError as exc:
            # None
            print("Type Error Detected, using already instantiated x_actual")
            self.predictions = self.model.predict(self.x_actual)
        return self.predictions
    
    # plot importance of features contribution in model, using the "gain"
    # metric
    def plot_importance(self, importance_type="gain", **kwargs):
        msg = "Importance can be specified only when base learner in XGBoost is'gbtree'!, Current model uses %s" %self.model.booster
        assert self.model.booster == "gbtree", msg
        xgb.plot_importance(self.model, importance_type=importance_type,
                title="Feature Importance by %s" %importance_type)

#*******************************************************************************
def plot_wall_test(dataframe, title, half_channel_grids, prediction=None,
        save=False,):
    """
    Function to make plot of test, shows plot of prediction if present in
    dataframe.
    Title: Title of Plot
    half_channel_grids: This is half of the grid sizes used in the simulation
    """
    means_u = return_average(dataframe, "u_plus")
    means_y = return_average(dataframe, "y_plus")

    with plt.style.context("seaborn-pastel"):
        fig, ax = plt.subplots()
        ax.plot(means_y[:half_channel_grids], means_u[:half_channel_grids],linestyle="-", label="Actual", color="red")
        if prediction:
            means_pred = return_average(dataframe, "predictions")
            ax.plot(means_y[:half_channel_grids],
                    means_pred[:half_channel_grids],linestyle="--",label="Predictions", color="blue")

    ax.set_xscale("log")
    ax.set_ylabel(r'$U^+$')
    ax.set_xlabel(r'$y^+$')
    ax.grid(True)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    if save:
        fig.savefig("%s.png" %title)
    else:
        plt.show()

def return_average(dataframe, column_name):
    return dataframe.groupby("Points:1")[column_name].mean().values

# Function to plot predictions against actual response values
def plot_predictions(dataframe, colormap=False):
    """
    Input: Dataframe with predictions and actual values"
    Output: Plot of predictions against actual response values, optional
    (colormap of difference)
    If colormap set to "True", difference between predictions and actual
    response values are plotted as well. (Takes longer for this operation)
    """

    # check that dataframe has columns labelled as required
    msg = "Column names should be PREDICTIONS and ACTUALS respectively!"
    assert "PREDICTIONS" and "ACTUALS" in dataframe.columns, msg
    fig, ax = plt.subplots()
    if colormap:
        dataframe['delta'] = np.abs(dataframe.ACTUALS - dataframe.PREDICTIONS)
        dataframe.plot.scatter(ax=ax , x="ACTUALS", y="PREDICTIONS", alpha=0.5,
                c='delta',colormap='jet')
    else:
        dataframe.plot.scatter(ax=ax , x="ACTUALS", y="PREDICTIONS", alpha=0.25)
    
    #add diagonal line on plot
    line = mlines.Line2D([0, 1], [0, 1], color='red')
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    ax.set_xlabel(r'$U^+$ (actual)')
    ax.set_ylabel(r'$U^+$ (predicted)')
    ax.set_title('Predictions vs Actual Values');


#**************************************************************************************

class XGBoost_Model(object):
    """
    Class for building a model using xgboost algorithm
    Model can be tuned as well
    """
    def __init__(self, params, dataframe):
        self.params = params
        self.dataframe = dataframe
        self.model = xgb.XGBRegressor(**params)
        self.train_data, self.validation_data = train_test_split(self.dataframe, test_size=0.3, random_state=100)
        train_x, train_y = cd.X_Y_split(self.train_data)
        validation_x, validation_y = cd.X_Y_split(self.validation_data)
        self.dtrain = xgb.DMatrix(data=train_x, label=train_y, feature_names=train_x.columns)
        self.dvalidation = xgb.DMatrix(data=validation_x, label=validation_y, feature_names=validation_x.columns)
        self.eval_matrix  = [(self.dtrain,"train"),(self.dvalidation,"validation")]
        self.eval_set = [(train_x,train_y),(validation_x,validation_y)]
    
    def __call__(self):
        return self.model
        
    def train_model(self, num_rounds=50, parameters=None, plot=True):
        if parameters != None:
            self.params.update(parameters)
        evals_result = {}
        self.model = xgb.train(params =self.params, dtrain=self.dtrain, 
                                       num_boost_round=num_rounds, early_stopping_rounds=50, evals=self.eval_matrix, verbose_eval=5, evals_result=evals_result)
        #if plot:
           # plot_fit(self)


    def add_evalset(self,dataframe):
        """
        Function to add additional dataset for validation during training. 
        dataframe must be cleaned before with clean_dataframe function.
        """
        dataframe = cd.clean_dataframe(dataframe)
        new_val_x, new_val_y = cd.X_Y_split(dataframe)
        new_val_mat = xgb.DMatrix(data=new_val_x, label=new_val_y, feature_names=new_val_x.columns)
        self.eval_matrix.append((new_val_mat, "validation_2"))
        self.eval_set.append((new_val_x, new_val_y))

    def predict(self, test_X):
        dtest = xgb.DMatrix(test_X, feature_names=test_X.columns)
        return self.model.predict(dtest)

    def fit(self, x=None, y=None, n_estimators=100, plot=True, save_plot=False,save_path=None):
        if x != None:
            train_x, train_y = (x, y)
        else:
            train_x, train_y = self.eval_set[0]
        self.model.set_params(**{"n_estimators":n_estimators})
        #call_backs =[ xgb.callback.print_evaluation(period=2)]
        self.fitted_model=self.model.fit(train_x, train_y,
                eval_set=self.eval_set , eval_metric="rmse",
                early_stopping_rounds=50)
        if plot:
            plot_fit(self.model, save=save_plot, save_path=save_path)
        return self.fitted_model

    def get_params(self):
        return self.params
    
    def set_params(self, parameters):
        """
        Parameters should be a list of parameters to update model with.
        """
        self.params.update(parameters)
        """
    def get_tuned_model(self,fit_data, parameters):
        tune_results = tune_parameter(self.model, fit_data, parameters)
        self.tuned_model = tune_results.best_estimator_
        best_params = tune_results.best_params_
        return self.tuned_model
        """

    def tune_model_parameter(self, parameter,  param_range, save_plot=False, save_path=None, randomized=True, n_iter = None, n_splits=5, data_size=0.1, fit_param=False):
        if data_size:
            tune_data = self.train_data.sample(frac=data_size)
        results = tune_parameter(tune_data, parameter,param_range, save_plot=save_plot, save_path=save_path, randomized=randomized, n_iter=n_iter, n_splits=n_splits, estimator=self.model)
        if fit_param:
            self.set_params(results.best_params_)
            self.model.set_params(**results.best_params_)
        return results

    def tune_all_parameters(self, param_distribution, n_iter=10, cv=5, data_size=0.1):
        tune_data = self.train_data.sample(frac=data_size)
        results = tune_all(tune_data,self.model,param_distribution, n_iter=n_iter, n_splits=cv)


# Function for tuning model using some parameters
"""
def tune_parameter(model, data, parameters, n_splits=5):
    train_x = data[0]
    train_y = data[1]
    #param_grid = dict(parameter=parameter_range)
    kfold = KFold(n_splits=n_splits, random_state=10)
    grid_search = GridSearchCV(model, parameters, verbose=0,
                              n_jobs=-1, cv=kfold, scoring="neg_mean_squared_error")
    grid_result = grid_search.fit(train_x, train_y, verbose=0)
    return grid_result
"""

def tune_parameter(data, parameter, param_range, save_plot=False, randomized=False, save_path=None, n_iter = None, n_splits=5,estimator=None):
    """
    Function to tune a parameter using either gridsearch or randomized search with possibility of cross validation.
    Input:
        - data = dataset to be used tuning, usually the training dataset.
        - parameter = string of parameter to be tuned. (works with XGBoost for now)
        - param_range = parameter search space
        - estimator = model to be tuned if existing already, if not a new default XGBRegressor model wil be created
    """

    train_x, train_y = cd.X_Y_split(data)
    param_grid = {parameter : list(param_range)}
    kfold = KFold(n_splits=n_splits, random_state=7)
    if not estimator:
        estimator = xgb.XGBRegressor(objective="reg:squarederror", )
    if randomized:
        assert n_iter != None, "Missing number of iterations"
        param_search = RandomizedSearchCV(estimator,param_grid,n_iter=n_iter,scoring="neg_mean_squared_error",cv=kfold)
    else:
        param_search = GridSearchCV(estimator, param_grid, verbose=0, cv=kfold, scoring="neg_mean_squared_error")
    grid_result = param_search.fit(train_x, train_y, verbose=0)
    
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    #for mean, sdev, param in zip(means, stds, params):
        #print("%f (%f) with: %r" % (mean, stdev, param))
        
    if randomized:
        param_range = [list(i.values())[0] for i in params]
        

    fig, ax = plt.subplots()
    ax.errorbar(param_range,-1*means, yerr=stds)
    ax.set_title("XGBoost %s vs RMSE" %parameter) 
    ax.set_xlabel('%s' %parameter)
    ax.set_ylabel('RMSE')
    if save_plot:
        if save_path:
            fig.savefig("%s/%s.png" %(save_path,parameter))
        else:
            fig.savefig("%s.png" %parameter)
    return grid_result

def plot_fit(model,train_results=None, save=False, save_path=None):
    """
    Function to plot train and test(validation) results of a training or a model fitting
    Input:
        - Results of fit or training
    """
    # plot evolution of training and validation results
    if train_results:
        results =train_results
    else:
        results = model.evals_result()
    epochs = len(results['validation_0']['rmse'])
    x_axis = range(0, epochs)

    # plot RMSE
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['rmse'], label='Train', c="b")
    ax.plot(x_axis, results['validation_1']['rmse'], label='Validation', c="r")
    ax.set(title=" Train and Validation RMSE")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("RMSE")
    ax.legend()
    if save:
        if save_path:
            fig.savefig("%s/training_evolution.png"%save_path)
        else:
            fig.savefig("model_evolution.png")


def tune_all(data, estimator, param_grid, n_iter=10,n_splits=5):
    train_x, train_y = cd.X_Y_split(data)
    kfold = KFold(n_splits=n_splits)
    param_search = RandomizedSearchCV(estimator,param_grid,n_iter=n_iter,
                                          scoring="neg_mean_squared_error",cv=kfold)
    grid_result = param_search.fit(train_x, train_y, verbose=0)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    return grid_result
