"""
Functions to display results of model after training.
"""

#import libraries
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
from sklearn.model_selection import cross_val_score

def display_results(model,train_data, validation_data, save=False, cv=False):
    """
    Current implement assumes training and validation data are supplied by
    default.
    Model type should be returned after training and having eval_results_
    """
    # unpack dataset
    train_x, train_y = train_data
    validation_x, validation_y = validation_data

    # calculate scores of model
    r_2_train = model.score(train_x, train_y)
    rmse_train = min(model.evals_result()['validation_0']['rmse'])
    if cv:
        r2_scores= cross_val_score(model, validation_x, validation_y, cv=10)
        r2_validation = np.mean(r2_scores)
        mse_scores = cross_val_score(model, validation_x, validation_y, scoring="neg_mean_squared_error", cv=10)
        mse_validation = np.mean(mse_scores)
        rmse_validaton = np.mean(np.sqrt(-1*mse_scores))
    else:
        r_2_validation = model.score(validation_x, validation_y) 
        rmse_validation = min(model.evals_result()['validation_1']['rmse'])
    
    # print some score results of the model
    print("**AFTER  TRAINING**\n")
    print("R_squared:")
    print(f"\ton Training set: {r_2_train :.3f}")
    print(f"\ton Validation set: {r_2_validation :.3f}\n")
    print("RMSE: ")
    print(f"\ton Training set: {rmse_train :.3f}")
    print(f"\ton Validation set: {rmse_validation :.3f}")

    # plot feature importance
    fig, ax = plt.subplots()

    feature_plot = xgb.plot_importance(model, ax =ax, importance_type="gain", title="Feature Importance after  Training", show_values=False)
    if save:
        fig.savefig('feat_importance_1.png')


    # plot evolution of training and validation results
    results = model.evals_result()
    epochs = len(results['validation_0']['rmse'])
    x_axis = range(0, epochs)

    # plot RMSE
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['rmse'], label='Train', c="b")
    ax.plot(x_axis, results['validation_1']['rmse'], label='Test', c="r")
    ax.set(title=" Training - Train and Validation RMSE")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("RMSE")
    ax.legend()
    if save:
        fig.savefig("Training - Evolution.png")

    # plot residuals
    predicted_train = model.predict(train_x)
    residuals_train = train_y.values.ravel() - predicted_train
    predicted_validation = model.predict(validation_x)
    residuals_validation = validation_y.values.ravel() - predicted_validation

    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
    ax[0].scatter(train_y.values.ravel(), residuals_train,c='b', alpha=0.5, s=40, cmap="coolwarm")
    ax[1].scatter(validation_y.values.ravel(), residuals_validation,c='b', alpha=0.5, s=40, cmap="coolwarm")
    ax[0].axhline(lw=2, color="red")
    ax[1].axhline(lw=2, color="red")
    ax[0].set_xlabel("Observed")
    ax[1].set_xlabel("Observed")
    ax[0].set_ylabel("Residual")
    ax[0].set(title="Training")
    ax[1].set(title="Validation")
    fig.suptitle("Fitted Values vs Residuals")
    if save:
        fig.savefig("Training-Residual_Analysis.png")

    # plot predicted values against actuals
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
    ax[0].scatter(x=train_y,y=predicted_train, alpha=0.5,)
    ax[1].scatter(x=validation_y,y=predicted_validation, alpha=0.5,)
    # draw diagonal lines through figure
    line_0 = mlines.Line2D([0, 1], [0, 1], color='red')
    line_1 = mlines.Line2D([0, 1], [0, 1], color='red')
    transform_0 = ax[0].transAxes
    transform_1 = ax[1].transAxes
    line_0.set_transform(transform_0)
    ax[0].add_line(line_0)
    line_1.set_transform(transform_1)
    ax[1].add_line(line_1)
    ax[0].set_xlabel(r'$U^+$ (actual)')
    ax[1].set_xlabel(r'$U^+$ (actual)')
    ax[0].set_ylabel(r'$U^+$ (predicted)')
    ax[0].set_title("Train")
    ax[1].set_title("Validation")
    fig.suptitle('Predictions vs Actual Values')
    if save:
        fig.savefig(" Training-Prediction_Analysis.png")

