import os, sys, glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utility.debug_tools import assert_colorize, pwc


def plot_data(data, x, y, outpath, tag):
    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
        # data = data[data.Timing == 'Train'].drop('Timing', axis=1)

    dir_name, file_name = os.path.split(outpath)
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    sns.set(style="whitegrid", font_scale=1.5)
    if 'Timing' in data.columns:
        sns.lineplot(x=x, y=y, ax=ax, data=data, hue=tag, style='Timing')
    else:
        sns.lineplot(x=x, y=y, ax=ax, data=data, hue=tag)
    ax.grid(True, alpha=0.8, linestyle=':')
    ax.legend(loc='best').set_draggable(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig(outpath)


def get_datasets(filedir, tag, condition=None):
    unit = 0
    datasets = []
    for root, _, files in os.walk(filedir):
        if 'log.txt' in files:
            log_path = os.path.join(root, 'log.txt')
            data = pd.read_csv(log_path, sep='\t')

            data.insert(len(data.columns), tag, condition)

            datasets.append(data)
            unit +=1

    return datasets

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    parser.add_argument('--outname', '-o')
    parser.add_argument('--legend', nargs='*')
    parser.add_argument('--legendtag', '-t', default='Algo')
    parser.add_argument('--x', default='Iteration', nargs='*')
    parser.add_argument('--y', default='ScoreMean', nargs='*')
    args = parser.parse_args()

    # by default assume using `python utility/plot.py` to call this file
    if len(args.logdir) != 1:
        dirs = [f'logs/{d}' for d in args.logdir]
    else:
        dirs = glob.glob(f'logs/{args.logdir[0]}/logs/GS-*')

    # set up legends
    if args.legend:
        assert_colorize(len(args.legend) == len(dirs),
            "Must give a legend title for each set of experiments.")
        legends = args.legend
    else:
        legends = [os.path.basename(path) for path in dirs]
        legends = [l[3:] if l.startswith('GS-') else l for l in legends]
    tag = args.legendtag

    pwc('Directories:')
    for d in dirs:
        pwc(f'\t{d}')
    pwc('Legends:')
    for l in legends:
        pwc(f'\t{l}')
    data = []
    for logdir, legend_title in zip(dirs, legends):
        data += get_datasets(logdir, tag, legend_title)

    if isinstance(args.x, list) or isinstance(args.y, list):
        xs = args.x if isinstance(args.x, list) else [args.x]
        ys = args.y if isinstance(args.y, list) else [args.y]
        for x in xs:
            for y in ys:
                outpath = f'results/{args.outname}/{x}-{y}.png'
                plot_data(data, x, y, outpath, tag)
    else:
        outpath = f'results/{args.outname}.png'
        plot_data(data, args.x, args.y, outpath, tag)

if __name__ == '__main__':
    main()