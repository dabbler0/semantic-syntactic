from lstms import *

model_prop = lstm_prop(sentence_prop)

large1 = Dataset('large-1')
model = large1[model_prop]
