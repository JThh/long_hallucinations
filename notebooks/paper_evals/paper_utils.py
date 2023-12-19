import os
import pickle
import yaml
import json

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from pprint import pprint
from deepdiff import DeepDiff

from copy import deepcopy
import seaborn as sns

from IPython.display import display

import wandb
api = wandb.Api()
api.entity = 'goatml'

# export to eval file
def get_vals(x):
    return x['mean'], x['mean']-x['bootstrap']['std_err'], x['mean']+x['bootstrap']['std_err']
    # return x['mean'], x['bootstrap']['low'], x['bootstrap']['high']

def get_vals2(x):
    return x['means'], x['low'], x['high']


def get_perfs(metrics):
    data = []
    for method in metrics['performance']:
        mean, low, high = get_vals(metrics['performance'][method])
        data.append([method, mean, low, high])
    
    df = pd.DataFrame(data, columns=['method', 'means', 'low', 'high'])
    return df.set_index('method')

def plot_bar(df, color='C0', ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    df.plot.bar(y='means', yerr=[df.means - df.low, df.high - df.means], color=color, ax=ax)

def get_uncertainty_df(metrics):
    data = []
    for method in metrics['uncertainty']:
        for metric in metrics['uncertainty'][method]:
            mean, low, high = get_vals(metrics['uncertainty'][method][metric])
            data.append([method, metric, mean, low, high])
    
    df = pd.DataFrame(data, columns=['method', 'metric', 'means', 'low', 'high'])
    
    df = df.set_index('method')

    def get_base_method(x):
        if 'p_false' in x:
            return 'p_false'
        elif 'p_ik' in x:
            return 'p_ik'
        elif 'regular_entropy' in x:
            return 'regular_entropy'
        elif 'cluster_assignment' in x:
            return x
        elif 'semantic_entropy' in x:
            return 'semantic_entropy'
        else:
            return x
    df['base_method'] = df.index.map(get_base_method)
    return df

def plot_bar_uncertainty(pdf, metric, remove='_UNANSWERABLE', ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    tmp = pdf
    # filter out unanswerable
    if remove is not None:
        tmp = tmp[tmp.index.map(lambda x: remove not in x)]
    tmp = tmp.reset_index()
    tmp = tmp.set_index('metric').loc[metric]
    tmp = tmp.set_index('method')
    tmp = tmp.sort_values('base_method')

    b2c = {m:f'C{i}' for i, m in enumerate(tmp.base_method.unique())}
    basemethod2color = lambda x: b2c[x]

    plot_bar(tmp, color=tmp.base_method.map(basemethod2color), ax=ax)
    ax.grid(axis='y', zorder=-10, alpha=0.3)



def restore_file(wandb_id, filenames=['wandb-summary.json', 'config.yaml']):
    files_dir = 'notebooks/restored_files'    
    os.system(f'mkdir -p {files_dir}')

    run = api.run(f'goatml/semantic_uncertainty/{wandb_id}')
    # run = api.run(f'goatml/uncertainty/{wandb_id}')
    out = []

    for filename in filenames:
        
        path = f'{files_dir}/{filename}'
        os.system(f'rm -rf {path}')
        run.file(filename).download(root=files_dir, replace=True, exist_ok=False)
    
        if filename.endswith('.pkl'):
            with open(path, 'rb') as f:
                out.append(pickle.load(f))
        elif filename.endswith('.yaml'):
            with open(path, 'r') as f:
                out.append(yaml.safe_load(f))
        elif filename.endswith('.json'):
            with open(path, 'r') as f:
                out.append(json.load(f))
        else:
            raise

    return out


def wandb_restore(wandb_run, filename):
    files_dir = 'tmp_wandb/'
    os.system(f'rm -rf {files_dir}')
    os.system(f'mkdir -p {files_dir}')

    run = api.run(wandb_run)
    run.file(filename).download(
        root=files_dir, replace=True, exist_ok=False)
    with open(f'{files_dir}/{filename}', 'rb') as f:
        out = pickle.load(f)
    return out, run.config



def dict_of_dfs_to_df(dictionary, key_name='wandbid'):

    keys = list(dictionary.keys())

    df = pd.DataFrame(dictionary[keys[0]])
    df[key_name] = keys[0]
    
    for key in keys[1:]:
        tmp = pd.DataFrame(dictionary[key])
        tmp[key_name] = key
        
        df = pd.concat([df, tmp])
    return df


def plot_grouped(plot_df, metric, x=None, hue='method', figsize=(10, 8), with_sort=True):
    if x is None:
        x = uniq_name
    fig, ax = plt.subplots(1, 1, figsize=figsize, sharey=True, dpi=100)

    if with_sort:
        plot_df = plot_df.sort_values([hue, x], ascending=True)
    plot_df = plot_df.fillna(0)
    plot_df[hue] = plot_df[hue].map(str)
    
    g1 = sns.barplot(x=x, y='means', hue=hue, data=plot_df, ax=ax)
    g2 = sns.barplot(x=x, y='low', hue=hue, data=plot_df, ax=ax, alpha=.5, edgecolor='lightgrey', fill=False, legend=False)
    g2 = sns.barplot(x=x, y='high', hue=hue, data=plot_df, ax=ax, alpha=.5, edgecolor='lightgrey', fill=False, legend=False)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90);
    ax.legend(loc=(1, 0.25), title=hue)
    ax.set_ylabel(metric)


def colorize(text, color=0):
    i2c = {i: f"\x1b[{a}" for i, a in enumerate([
    '1m', '31m', '33m', '34m', '35m', '36m', '37m', '38m'])}
    r = "\x1b[0m"

    return i2c[color] + text + r


def get_perf_df(runs, all_results):
    perfdf = {}
    for wandb_id, run_name in runs.items():
        results = all_results[wandb_id]    
        perfdf[wandb_id] = get_perfs(results)
    
    runids = list(perfdf.keys())
    tmp = pd.DataFrame(perfdf[runids[0]])
    tmp['run'] = runs[runids[0]]
    
    for runid in runids[1:]:
        tmp2 = pd.DataFrame(perfdf[runid])
        tmp2['run'] = runs[runid]
        
        tmp = pd.concat([tmp, tmp2])
    
    perfdf = tmp.reset_index()
    
    tmp = perfdf.set_index('method').loc['accuracy']
    tmp['model'] = tmp.run.map(lambda x: '-'.join(x.split('-')[:1]))
    tmp['dataset'] = tmp.run.map(lambda x: '-'.join(x.split('-')[1:]))
    display(tmp.pivot(columns='model', index='dataset', values='means'))

    pdfs = {}
    for wandb_id in runs:
        results = all_results[wandb_id]
        pdfs[wandb_id] = get_uncertainty_df(results)
    
    runids = list(pdfs.keys())
    
    tmp = pd.DataFrame(pdfs[runids[0]])
    tmp['run'] = runs[runids[0]]
    tmp['wandb_id'] = runids[0]
    
    for runid in runids[1:]:
        tmp2 = pd.DataFrame(pdfs[runid])
        tmp2['run'] = runs[runid]
        tmp2['wandb_id'] = runid
    
        tmp = pd.concat([tmp, tmp2])
    
    all_runs = tmp.reset_index()

    return perfdf, pdfs, all_runs



def check_first_item(configs):
    # Manually make sure they all operate on the same random split of the data by reading the logs
    
    for wandbid, config in configs.items():
        slurmid = api.run(f'goatml/semantic_uncertainty/{wandbid}').notes.split(',')[0][len('slurm_id: '):]
        first_item = os.popen(f"grep -m 1 -A 10 'NEW ITEM' ../../log/*{slurmid}*").read().split('\n')
        try:
            q_line = np.where(['Question:' in l for l in first_item])[0][0]
            print(wandbid, slurmid, config['dataset']['value'], first_item[q_line + 1][28:])
        except:
            q_line = first_item
            if wandbid == 'e0z3555u':
                print(wandbid, slurmid, config['dataset']['value'], 'MANUAL: What act sets forth the functions of the Scottish Parliament?')
            else:
                print('Failure for', wandbid, slurmid)


from scipy.stats import sem 

stats = lambda x: {'mean': np.mean(x), 'sem': sem(x), 'std': np.std(x), 'median': np.median(x)}


def get_length_statistic(all_gens):
    # extract length of generations
    low_temp, high_temp = {}, {}
    for wandbid, gens in all_gens.items():
        low_temp[wandbid], high_temp[wandbid] = [], []
        for _, gen in gens[0].items():
            low_temp[wandbid].append(len(gen['most_likely_answer']['response']))
            high_temp[wandbid].extend([len(r[0]) for r in gen['responses']])

    stats = {'mean': np.mean, 'sem': sem, 'std': np.std, 'median': np.median}
    columns = ['temp', 'wandbid', 'statistic', 'value']
    
    data = []
    for temp_name, temp in zip(['low', 'high'], [low_temp, high_temp]):
        for wandbid, lengths in temp.items():
            for statname, statfunc in stats.items():
                data.append([temp_name, wandbid, statname, statfunc(lengths)])

    df = pd.DataFrame(data, columns=columns)    
    assert set(df.wandbid.unique()) == set(list(all_gens.keys()))
    
    return low_temp, high_temp, df



def get_cluster_stats(num_ids):
    stats = {'mean': np.mean, 'sem': sem, 'std': np.std, 'median': np.median}
    columns = ['wandbid', 'statistic', 'value']
    
    data = []
    for wandbid, counts in num_ids.items():
        for statname, statfunc in stats.items():
            data.append([wandbid, statname, statfunc(counts)])
    
    df = pd.DataFrame(data, columns=columns)    
    return df