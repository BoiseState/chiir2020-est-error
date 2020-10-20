import logging, logging.config
import yaml

import pystan


with open('logging.yaml') as lf:
    log_config = yaml.load(lf)
    
logging.config.dictConfig(log_config)
_log = logging.getLogger('pystan.runner')


pop_unif_model_code = """
data {
    int<lower=1> U; // number of users
    int<lower=1> I; // number of items
    int<lower=1> L; // length of the recommendation list
    int<lower=0,upper=1> X[U, I];
    int<lower=0> REC_LIST[U, L];
}
parameters {
    real<lower=0> mu[I];
    real<lower=0> sigma;
    real logit_pi[U, I];
    
}
transformed parameters {
    real<lower=0, upper=1> pi[U, I];
    pi = inv_logit(logit_pi);

}
model {
    mu ~ exponential(0.001);
    sigma ~ exponential(0.001);
    
    for (u in 1:U) {
        logit_pi[u] ~ normal(mu, sigma);
    }
    
    for (u in 1:U) {
        X[u] ~ bernoulli(pi[u]);
    }
    
}
generated quantities {
    real recall=0;

    for (u in 1:U) {
        real pi_l[L];
        real n_pi;
        real k_pi;
        
        n_pi = sum(round(pi[u]));
        
        for (i in 1:L) {
            int item_id = REC_LIST[u,i] + 1;
            pi_l[i] = pi[u, item_id];
        }
        k_pi = sum(round(pi_l));
        
        recall += n_pi / k_pi;
    }
    recall = recall / U;
}
"""


import pandas as pd


ratings = pd.read_csv('./data/ml-100k/u.data', sep='\t', names=['user', 'item', 'rating', 'timestamp']).drop('timestamp', axis=1)


recommendations = pd.read_csv('./build/recommendations_item-item.csv')


item_index = pd.Index(ratings.item.unique())
user_index = pd.Index(recommendations.user.unique())



from scipy.sparse import csr_matrix



sample_ratings = ratings[ratings['user'].isin(recommendations['user'].unique())]



import numpy as np



rows = user_index.get_indexer(sample_ratings['user'])
cols = item_index.get_indexer(sample_ratings['item'])
data = np.ones_like(rows)



nusers = len(user_index)
nitems = len(item_index)
print(nusers, nitems)



csr_ratings = csr_matrix((data, (rows, cols)), shape=(nusers, nitems))



dense_ratings = csr_ratings.toarray()



print(dense_ratings.shape)


# In[19]:


X = dense_ratings


# In[20]:


print(len(X), len(X[0]))


# In[22]:


recommendations['item_ind'] = item_index.get_indexer(recommendations['item'])
recommendations['user_ind'] = user_index.get_indexer(recommendations['user'])


# In[23]:


REC_LIST = recommendations[['user_ind', 'item_ind', 'rank']].pivot_table(index='user_ind', columns='rank', values='item_ind').values
L = len(REC_LIST[0])


# In[ ]:


model_data = {'U': nusers,
              'I': nitems,
              'L': L,
              'REC_LIST':REC_LIST,
              'X': X}

sm = pystan.StanModel(model_code=pop_unif_model_code)


# In[ ]:


import pickle as pkl


# In[ ]:


with open('build/pop_pref_unif_obs_model_1.pkl', 'wb') as f:
    pkl.dump(sm, f)
print('pop_pref_unif_obs_model.pkl saved')


# In[ ]:


fit = sm.sampling(data=model_data, iter=1000, chains=4, n_jobs=14)


# In[ ]:


la = fit.extract(permuted=True)


# In[ ]:


with open('build/pop_unif_model_sampling_1.pkl', 'wb') as f:
    pkl.dump(la, f)
print('pop_unif_model_sampling_1.pkl saved')

