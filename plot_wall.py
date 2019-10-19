"""
funtion to plot LOW from dataframe
inputs: 
    datarame - dataframe with details to be plotted. may or may not have predictions as well
    title - title of chart, or test
"""
import matplotlib.pyplot as plt

def plot_wall(dataframe, title):
    #sort dataframe based on y values
    if 'Points.1' in dataframe.columns:
        dataframe.rename(columns={'Points.1':'Points:1'}, inplace=True)
    dataframe = dataframe.sort_values(['Points:1'])
    y_set = sorted(list(set(dataframe['Points:1'])))
    # calculate averages for each delta
    for idx, y in enumerate(y_set):
        dataframe.loc[dataframe['Points:1'] == y_set[idx], 'avg_uplus_orig']= dataframe.loc[dataframe['Points:1'] == y_set[idx], 'u_plus'].mean(axis=0)
        dataframe.loc[dataframe['Points:1'] == y_set[idx], 'avg_yplus']= dataframe.loc[dataframe['Points:1'] == y_set[idx], 'y_plus'].mean(axis=0)
        if 'xgb_pred' in dataframe.columns:
            dataframe.loc[dataframe['Points:1'] == y_set[idx], 'avg_uplus_xgb']= dataframe.loc[dataframe['Points:1'] == y_set[idx], 'xgb_pred'].mean(axis=0)
        if 'lasso_pred' in dataframe.columns:
            dataframe.loc[dataframe['Points:1'] == y_set[idx], 'avg_uplus_lasso']= dataframe.loc[dataframe['Points:1'] == y_set[idx], 'lasso_pred'].mean(axis=0)
        if 'ridge_pred' in dataframe.columns:
            dataframe.loc[dataframe['Points:1'] == y_set[idx], 'avg_uplus_ridge']= dataframe.loc[dataframe['Points:1'] == y_set[idx], 'ridge_pred'].mean(axis=0)
    
    
    #plot law of the wall
    fig=plt.figure(1, figsize=(7.5, 5), dpi=80)
    
    #select only first 32 representing one half of channel
    to_plot_up = dataframe.loc[:32,'avg_uplus_orig']
    to_plot_yp = dataframe.loc[:32,'avg_yplus']
    try:
        to_plot_up_xgb = dataframe.loc[:32,'avg_uplus_xgb']
        plt.plot(to_plot_yp,to_plot_up_xgb,color ='blue', marker="v",linewidth=2.0)
    except:
        None
    #try:
    to_plot_up_ridge = dataframe.loc[:32,'avg_uplus_ridge']
    plt.plot(to_plot_yp,to_plot_up_ridge,color ='green', marker="v",linewidth=2.0)
    #except:
        #None
    
    try:
        to_plot_up_lasso = dataframe.loc[:32,'avg_uplus_lasso']
        plt.plot(to_plot_yp,to_plot_up_lasso,color ='blue', marker="v",linewidth=2.0)
    except:
        None

    #plt.plot(to_plot_yp,to_plot_up,color ='red',linewidth=1.0)
    plt.plot(to_plot_yp,to_plot_up_ridge,color ='green', marker="v",linewidth=2.0)
    
    plt.axis([1, 200, 0, 20])
    plt.xscale('log')
    plt.ylabel(r'$U^+$')
    plt.xlabel(r'$y^+$')
    plt.tight_layout()
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)
    plt.legend()
    plt.title(title)
    plt.show()