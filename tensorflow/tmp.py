#%%
import tensorflow as tf
#%%
s = tf.nn.embedding_lookup([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1]*120, [2]*120])
tf.Session().run(s)

#%%
m = tf.placeholder(dtype=tf.int32, shape=[None, 120], name="mask")
tf.Session().run(m)

#%%
import json

with open('./data/train.json') as json_file:  
    data = json.load(json_file)

i = 0
last_relfact = ''
for idx, ins in enumerate(data):
    cur_relfact = ins['head']['id'] + '#' + ins['tail']['id'] + '#' + ins['relation']
    if cur_relfact == last_relfact:
        print(cur_relfact)
        print(last_relfact)
        print(ins['sentence'])
        print(ins['head']['word'])
        print(ins['tail']['word'])
        print(data[idx-1]['sentence'])
        print(data[idx-1]['head']['word'])
        print(data[idx-1]['tail']['word'])
        i+=1
        if i == 1:
            break
    last_relfact = cur_relfact

#%%

import json
import pandas as pd

with open('./data/train.json') as json_file:  
    data = json.load(json_file)

computed = {}
for idx, ins in enumerate(data):
    if not computed.get(ins['relation']):
        computed[ins['relation']] = [1.0]
    else:
        computed[ins['relation']][0] += 1.0

df = pd.DataFrame(computed)

#%%
df.info()