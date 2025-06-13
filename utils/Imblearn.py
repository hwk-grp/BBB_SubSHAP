import numpy
from imblearn import FunctionSampler  # to use a idendity sampler
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours, RandomUnderSampler
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, RandomOverSampler, SVMSMOTE, SMOTE, KMeansSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE


def plot_resampling(X, y, sampler, ax, title=None):
    X_res, y_res = sampler.fit_resample(X, y)
    tsne_np = TSNE(n_components=2, n_iter=1000, random_state=10).fit_transform(X_res)

    ax.scatter(tsne_np[y_res==0, 0], tsne_np[y_res==0, 1] ,color="#e66101",marker="^" ,alpha=0.8, edgecolor="k",s=35)
    ax.scatter(tsne_np[y_res==1, 0], tsne_np[y_res==1, 1] ,color="#fdb863",marker="o" ,alpha=0.8, edgecolor="k",s=35)
    if title is None:
        title = f"{sampler.__class__.__name__}"
    ax.set_title(title,fontsize=16)
    sns.despine(ax=ax, offset=8)
    sns.set_style("ticks")

def return_emb_after_Imblearn(train_data, train_targets, rand_seed, data_kind):


    # Embedding vector space
    data_train_embedding = train_data
    data_train_target = numpy.array(train_targets).reshape(-1,1)
    data_train_target = data_train_target.astype('int64')

    X, y = data_train_embedding, data_train_target
    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(15, 15))


    samplers = [
        FunctionSampler(),
        SMOTE(random_state=0),
        RandomOverSampler(random_state=0),
        ADASYN(random_state=0),
        BorderlineSMOTE(random_state=0),
        SVMSMOTE(random_state=0),
        KMeansSMOTE(random_state=0),
        RandomUnderSampler(random_state=0),
        TomekLinks(),
        EditedNearestNeighbours(),
        SMOTETomek(smote=SMOTE(random_state=0)),
        SMOTEENN(smote=SMOTE(random_state=0)),
    ]

    for ax, sampler in zip(axs.ravel(), samplers):
        title = "Original dataset" if isinstance(sampler, FunctionSampler) else None
        plot_resampling(X, y, sampler, ax, title=title)

    fig.tight_layout()
    fig.savefig('preds/draws/dataplot_seed' + str(rand_seed) + '_input' + data_kind + '.pdf', dpi=1000)


    list_emb = []

    for ax, sampler in zip(axs.ravel(), samplers):
        X_res, y_res = sampler.fit_resample(X, y)
        X_res = X_res.tolist()
        for i in range(0, len(X_res)):
            X_res[i].append(y_res[i])
        list_emb.append(X_res)

    return list_emb
