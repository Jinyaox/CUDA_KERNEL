import matplotlib.pyplot as plt
import seaborn
import pandas as pd

def plot_density(df,size,ax):
    dfs=df.query("size == {}".format(str(size)))
    seaborn.lineplot(data=dfs,x='density', y='time', hue='method',style="method",ax=ax)
    ax.set_yscale("log")

def plot_size(df,density,ax):
    dfs=df.query("density == {}".format(str(density)))
    seaborn.lineplot(data=dfs,x='size', y='time', hue='method',style="method",ax=ax)
    ax.set_yscale("log")

if __name__ == '__main__':
    df=pd.read_csv("result.csv")
    fig, axs = plt.subplots(1, 2)
    plot_density(df,4096,axs[0])
    plot_size(df,0.015625,axs[1])
    plt.savefig('result.png')
