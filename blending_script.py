import numpy as np, pandas as pd

m1 = 'sub/kg.csv'
# m2 = 'sub/lr_submission.csv'
# m3 = 'sub/submission_cln_lr.csv'
m4 = 'sub/seed_4.csv'

f1 = pd.read_csv(m1)
# f2 = pd.read_csv(m2)
# f3 = pd.read_csv(m3)
f4 = pd.read_csv(m4)

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
p_res = f1.copy()
p_res[label_cols] = f1[label_cols]*0.40 + f4[label_cols]*0.60#) / 2
p_res.to_csv('sub/ens_submission.csv', index=False)


