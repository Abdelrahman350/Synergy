import os
from os import path
import pandas as pd
import matplotlib.pyplot as plt

def plot_history(model_path, model_name):
    def plot_feature_loss(feature):
        fig = plt.figure()
        save_dir = os.path.join(model_plots_dir, feature)
        if feature in ['Pm_avg', 'loss', 'Pm_pitch', 'Pm_roll', 'Pm_yaw']:
            feature_loss = feature
            val_feature_loss = 'val_'+feature
        else:
            feature_loss = feature+'_loss'
            val_feature_loss = 'val_'+feature+'_loss'
        plt.plot(data['epoch'], data[feature_loss])
        plt.plot(data['epoch'], data[val_feature_loss])
        plt.xlabel('epochs')
        plt.ylabel(feature)
        plt.grid()
        plt.legend([feature_loss, val_feature_loss], loc="best")
        plt.savefig(save_dir)
        plt.close()
        
    csv_path = os.path.join(model_path, model_name.strip('.')+'.csv')
    data = pd.read_csv(csv_path)
    history_dir = 'history'
    model_plots_dir = os.path.join(history_dir, model_name)
    if not path.exists(history_dir):
        os.makedirs(history_dir)
    if not path.exists(model_plots_dir):
        os.makedirs(model_plots_dir)
    plot_feature_loss('Pm')
    plot_feature_loss('Pm*')
    plot_feature_loss('Lc')
    plot_feature_loss('Lr')
    plot_feature_loss('loss')
    plot_feature_loss('Pm_pitch')
    plot_feature_loss('Pm_yaw')
    plot_feature_loss('Pm_roll')
    plot_feature_loss('Pm_avg')
