# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
# import matplotlib
import errno
import os
import ntpath


# import os.path as osp
# import datetime
# import glob

# matplotlib.use('TkAgg') # Can change to 'Agg' for non-interactive mode
#
# import matplotlib.pyplot as plt
# plt.rcParams['svg.fonttype'] = 'none'

# # def plot_trading(df, dirname, i, action):
def data_p(path):
    df = pd.read_csv(path)
    if not df.empty:
        pre, ext = os.path.splitext(path)
        spath, s_num = ntpath.split(pre)
        try:
            spath = os.path.join(spath, "images")
            os.makedirs(spath)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise  # This was not a "directory exist" error..
        file_name = os.path.join(spath, s_num + '.png')

        plot_trading(df, file_name)
    else:
        print("empty dataframe, removing ")
        os.remove(path)

def plot_trading(df, save_path):
    price = [100]
    for perc in df['p1'][1:]:
        price.append(price[0] * (1 + perc))
    df['price'] = price

    cr = [0]
    for rew in df['r']:
        cr.append(cr[-1] + rew - 0.5)
    df['Cum_rew'] = cr[:-1]

    fig, host = plt.subplots()
    fig.subplots_adjust(right=0.75)

    par1 = host.twinx()

    tkw = dict(size=4, width=1.5)

    df['Signal'] = df['a'] - df['a'].shift(1)
    df['Signal'].iat[0] = df['a'].iloc[0]

    # p3 = host.scatter(df.index[df['Actions']==0], df['Observations'][df['Actions']==0], marker=".", color = 'b')
    p3 = host.scatter(df.index[(df['Signal'] < 0) & (df['a'] != 0)], df['price'][(df['Signal'] < 0) & (df['a'] != 0)], marker="v", color='r', zorder=2)
    p3 = host.scatter(df.index[(df['Signal'] > 0) & (df['a'] != 0)], df['price'][(df['Signal'] > 0) & (df['a'] != 0)], marker="^", color='g', zorder=2)
    p3 = host.scatter(df.index[(df['Signal'] != 0) & (df['a'] == 0)], df['price'][(df['Signal'] != 0) & (df['a'] == 0)], marker="o", color='k', zorder=2)

    p3, = host.plot(df['price'], "b-", label="Stock price", zorder=1)
    p1, = par1.plot(df['Cum_rew'], "m-", label="Cumulated Rewards")

    # host.set_xlim(0, 2)
    host.set_ylim(99.8, 100.2)
    # par1.set_ylim(-1, 1)
    # par2.set_ylim(0, 0.1)

    host.set_xlabel("Timesteps")
    host.set_ylabel("Stock price")
    par1.set_ylabel("Cumulated Rewards")

    par1.yaxis.label.set_color(p1.get_color())
    host.tick_params(axis='y', colors=p3.get_color(), **tkw)
    host.ticklabel_format(useOffset=False)
    host.yaxis.label.set_color(p3.get_color())

    par1.tick_params(axis='y', colors=p1.get_color(), **tkw)
    host.tick_params(axis='x', **tkw)
    # plt.ticklabel_format(axis="y", style="plain", scilimits=(0, 0))
    xint = range(0,df.shape[0]+1)
    plt.xticks(xint)
    # plt.show()
    # folder= os.path.dirname(os.path.abspath(path))
    # save_path
    plt.savefig(os.path.join(save_path))

    # plt.savefig(os.path.join(folder, "results_plot.png"))
    plt.close()

if __name__ == '__main__':

    # data = pd.read_csv("/Users/edoardovittori/Code/alphazero_singleplayer/logs/Trading-v0/2020-07-13_11-35-43/state_action/11793779510439986209.csv")
    # price = [100]
    # for perc in data['p1'][1:]:
    #     price.append(price[0] * (1 + perc))
    # data['price']=price
    #
    # cr = [0]
    # for rew in data['r']:
    #     cr.append(cr[-1]+rew-0.5)
    # data['Cum_rew']=cr[:-1]
    #
    # print(data)
    # path = "/Users/edoardovittori/Code/alphazero_singleplayer/logs/Trading-v0/2020-07-13_14-40-22/state_action/3261920518430412717.csv"
    path = "/Users/edoardovittori/Code/alphazero_singleplayer/logs/Trading-v0/2020-07-13_14-56-33/state_action/11204837701962185700.csv"
    data_p(path)

