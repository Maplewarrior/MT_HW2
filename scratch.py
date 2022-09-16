train_model(ipt, epochs)             
EMB = Embedder(vocab_size, d_model)
e1 = EMB(ipt['input_ids'][0].unsqueeze(1))
f1 = EMB(ipt['target'][0].unsqueeze(1))

# 
# p_e1 = pe(e1)


# print("##### Example shown here #####")
# EMB = Embedder(vocab_size, d_model)

# e1 = df_small['en'][0]
# e2 = df_small['en'][2]
# f1 = df_small['fr'][0]

# e = tokenizer(e1, return_tensors='pt')
# e1 = tokenizer(e1, return_tensors='pt')
# e2 = tokenizer(e2, return_tensors='pt')

# # print(e1)

# e1 = EMB.forward(e1['input_ids'])
# p_e1 = pe(e1)


print("##### Example shown here #####")
EMB = Embedder(vocab_size, d_model)

e1 = df_small['en'][0]
e2 = df_small['en'][2]
f1 = df_small['fr'][0]

e = tokenizer(e1, return_tensors='pt')
e1 = tokenizer(e1, return_tensors='pt')
e2 = tokenizer(e2, return_tensors='pt')

# print(e1)

e1 = EMB.forward(e1['input_ids'])
p_e1 = pe(e1)


# f1 = tokenizer(f1, return_tensors='pt')

# print("embedding")
# # e1 = EMB.forward(e1['input_ids'])
# e2 = EMB.forward(e2['input_ids'])
# f1 = EMB.forward(f1['input_ids'])

# print("f1:", f1.size())
# out_e = encoder.forward(e['input_ids'])
# #%%

# print("positional encoding")
# # print(e1.size())

# e1 = pe.forward(e1)
# Q, K, V = e1, e1, e1

# out = MHA.forward(Q, K, V)

# out = FFN.forward(out)

# out_e1 = encoderLayer(out)
# print("encoderLayer out:\n",out_e1.size())



# out_d1 = decoderLayer1.forward(f1, out_e1, None, None)
