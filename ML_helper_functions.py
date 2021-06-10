from sklearn.inspection import permutation_importance
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import learning_curve
from sklearn.metrics import get_scorer
from sklearn.model_selection import train_test_split, cross_val_score

from copy import copy

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

dpi = 200


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, scoring='accuracy', train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate test and training learning curve
    """
    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(6, 6))

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training set size")
    axes.set_ylabel("Score: %s" % scoring)

    train_sizes, train_scores, test_scores = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, scoring=scoring,
                       train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes.legend(loc="best")
    
    plt.show()

    return axes









def train_test_estimate(X, y, model, scorer):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=23)
    model.fit(X_train, y_train)
    score = scorer(model, X_test, y_test)
    return score


def y_randomization_test(yx, model_dicts, crys_prop, scoring='f1_macro', n_repeats=25, savefig=False):

    fig, ax = plt.subplots(nrows=len(model_dicts[crys_prop]), ncols=1,
                           figsize=(6.5, len(model_dicts[crys_prop])*3),
                           sharex=True)

    scorer = get_scorer(scoring)

    for i, (model_name, model) in enumerate(model_dicts[crys_prop].items()):

        X = yx.iloc[:, 1:].copy()
        y = yx.iloc[:, 0].astype(int).copy()

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=23)
        m = copy(model)
        y_initial_score = scorer(m.fit(X_train, y_train), X_test, y_test)

        y_randomized_scores = []
        for jj in range(n_repeats):
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=23+jj)
            m = copy(model)

            np.random.seed(92)
            np.random.shuffle(y_train)

            score_y_randomized = scorer(m.fit(X_train, y_train), X_test, y_test)
            y_randomized_scores.append(score_y_randomized)


        #y_initial_score = train_test_estimate(X, y, copy(model), scorer)

        #y_randomized_scores = []
        #for _ in range(n_repeats):

        #    np.random.seed(92)
        #    np.random.shuffle(y)
        #    score_y_randomized = train_test_estimate(X, y, copy(model), scorer)
        #    y_randomized_scores.append(score_y_randomized)

        ax[i].hist(y_randomized_scores, bins=15, color='blue', label='y-randomized')
        ax[i].axvline(y_initial_score, color='red', lw=4, label='y-initial')
        ax[i].set_title('%s model' % (model_name))
        ax[i].legend(loc='upper left')
        ax[i].set_xlim(0.0, 1.0)

    ax[len(model_dicts[crys_prop])-1].set_xlabel(scoring)

    fig.suptitle(f'$\Delta${crys_prop} prediction', y=.995, fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    if savefig:
        fig.savefig(f'y_randomization_test_{crys_prop}.png', dpi=dpi)
    plt.show()










def show_importances(crys_prop, yx, model_dicts, savefig=False):
    
    model_importances = []

    for m_name, m in model_dicts[crys_prop].items():

        result = permutation_importance(estimator=m,
                                        X=yx.iloc[:, 1:], y=yx.iloc[:, 0].astype(int),
                                        scoring='f1_macro',
                                        n_repeats=10,
                                        random_state=23)

        model_importances.append([m_name, result.importances_mean])
    
    # https://github.com/bhimmetoglu/feature-selection/blob/master/FeatSelection.ipynb
    all_i = pd.DataFrame(data=np.hstack([v[1].reshape(-1, 1) for v in model_importances]),
                         index=yx.columns[1:],
                         columns=[v[0] for v in model_importances])

    all_i = all_i.drop(columns='DUM')

    annot_array = all_i.T.copy()
    all_i = all_i.apply(minmax_scale, axis=0)
    df_ = all_i.T

    plt.rc('ytick', labelsize=20)
    fig = plt.figure(figsize=(14, 7))
    plt.rc('xtick', labelsize=18)

    cmap = sns.color_palette('bone_r', df_.shape[1])
    sns.heatmap(df_, cmap=cmap,
                annot=annot_array, annot_kws={'fontsize': 18}, 
                fmt='.2g',
                vmax=1.0, vmin=0,
                linewidths=.5,
                cbar=False, cbar_kws={'label': 'relative feature importance'})
    plt.title('Permutation feature importances for models predicting $\Delta$%s' % crys_prop)
    plt.xticks(rotation=45)
    
    if savefig:
        fig.tight_layout()
        fig.savefig(f'permutation_FI_{crys_prop}_summary.png', dpi=dpi)

    plt.show()


def show_importance_hists(crys_prop, yx, model_dicts, savefig=False):

    fig, ax = plt.subplots(len(model_dicts[crys_prop]), 1, figsize=(12, len(model_dicts[crys_prop])*4))

    for i, (m_name, m) in enumerate(model_dicts[crys_prop].items()):

        result = permutation_importance(estimator=m,
                                        X=yx.iloc[:, 1:], y=yx.iloc[:, 0].astype(int),
                                        scoring='f1_macro',
                                        n_repeats=10,
                                        random_state=23)

        perm_sorted_idx = result.importances_mean.argsort()

        ax[i].boxplot(result.importances[perm_sorted_idx].T,
                      vert=True,
                      labels=yx.columns[1:][perm_sorted_idx])
        ax[i].set_title('Permutation feature importances for %s model predicting $\Delta$%s' % (m_name, crys_prop))
        ax[i].tick_params(axis='x', labelsize=14, rotation=45)
        ax[i].yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
        ax[i].grid(color='k', linestyle=':', linewidth=1, axis='y')

    fig.tight_layout()
    if savefig:
        fig.savefig(f'permutation_FI_{crys_prop}_separate.png', dpi=dpi)
    plt.show()









def inhome_permutation_importance(estimator,
                                   feature_groups, X, y,
                                   scoring='f1_macro',
                                   n_repeats=10,
                                   random_state=23):

    result = {'score_difference': np.zeros((len(feature_groups), n_repeats)),
              'feature_group_names': np.zeros(len(feature_groups), dtype='O')}

    X_train_original, X_test, y_train, y_test = train_test_split(X, y,
                                                                test_size=.2,
                                                                stratify=y,
                                                                random_state=random_state)

    feature_list_indices = {col: idx for idx, col in enumerate(X_train_original.columns)}
    scorer = get_scorer(scoring)

    for i, (feature_group_name, feature_list) in enumerate(feature_groups.items()):
#         print(feature_group_name)

        score_difference = []

        for j in range(n_repeats):

            np.random.seed(random_state+j)
            X_train_permuted = X_train_original.copy()

            # permute feature values in selected columns
            for col in feature_list:
    #             print(col)
                col_idx = feature_list_indices[col]
                permuted_indices = np.random.permutation(X_train_original.shape[0])

#                 col = pd.DataFrame(np.random.uniform(low=-1.0, high=1.0, size=X_train_original.shape[0])) # fill with random values from U(-1, 1)
                col = X_train_permuted.iloc[permuted_indices, col_idx] # permute present values
                col.index = X_train_permuted.index
                X_train_permuted.iloc[:, col_idx] = col

#             X_train_permuted = X_train_permuted.drop(columns=feature_list)

            # train model using OLD data matrix X_train_original and evaluate
            est_original = estimator.fit(X_train_original, y_train)
            score_original = scorer(est_original, X_test, y_test)

            # train model using NEW data matrix X_train_permuted and evaluate
            est_permuted = estimator.fit(X_train_permuted, y_train)
            score_permuted = scorer(est_permuted, X_test, y_test)


            result['score_difference'][i, j] = score_original - score_permuted

        result['feature_group_names'][i] = feature_group_name

    return result


def feature_group_permutation_and_refit_importances(yx, model_dicts, crys_prop, feature_groups):

    fig, ax = plt.subplots(len(model_dicts[crys_prop]), 1, figsize=(7, len(model_dicts[crys_prop])*7))

    for i, (model_name, model) in enumerate(model_dicts[crys_prop].items()):

        result = inhome_permutation_importance(estimator=copy(model),
                                                feature_groups=feature_groups,
                                                X=yx.iloc[:, 1:], y=yx.iloc[:, 0].astype(int),
                                                scoring='f1_macro',
                                                n_repeats=10,
                                                random_state=23)

        perm_sorted_idx = result['score_difference'].mean(axis=1).argsort()

        ax[i].boxplot(result['score_difference'][perm_sorted_idx].T,
                      vert=True,
                      labels=result['feature_group_names'][perm_sorted_idx])
        ax[i].tick_params(axis='x', rotation=90)
        ax[i].set_title('%s model permutation feature importance' % model_name)

    fig.tight_layout()
    plt.show()