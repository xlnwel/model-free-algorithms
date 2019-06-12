import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utility.debug_tools import assert_colorize, pwc


def plot_data(data, value="AvgScore"):
    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)

    sns.set(style="darkgrid", font_scale=1.5)
    sns.tsplot(data=data, time="Iteration", value=value, unit="Unit", condition="Condition")
    plt.legend(loc='best').draggable()
    plt.show()


def get_datasets(filedir, condition=None):
    unit = 0
    datasets = []
    for root, exp_name, files in os.walk(filedir):
        if 'log.txt' in files:
            log_path = os.path.join(root, 'log.txt')
            pwc(f'log path: {log_path}')
            data = pd.read_table(log_path)

            data.insert(len(data.columns), 'Unit', unit)
            data.insert(len(data.columns), 'Condition', condition or exp_name)

            datasets.append(data)
            unit +=1

    return datasets

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    parser.add_argument('--legend', nargs='*')
    parser.add_argument('--value', default='AvgScore', nargs='*')
    args = parser.parse_args()

    use_legend = False
    if args.legend is not None:
        assert_colorize(len(args.legend) == len(args.logdir),
            "Must give a legend title for each set of experiments.")
        use_legend = True

    data = []
    if use_legend:
        for logdir, legend_title in zip(args.logdir, args.legend):
            data += get_datasets(logdir, legend_title)
    else:
        for logdir in args.logdir:
            pwc(data)
            data += get_datasets(logdir)

    if isinstance(args.value, list):
        values = args.value
    else:
        values = [args.value]
    for value in values:
        plot_data(data, value=value)

if __name__ == '__main__':
    main()