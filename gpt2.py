from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
lmmodel = GPT2LMHeadModel.from_pretrained('gpt2-medium').cuda()
model = GPT2Model.from_pretrained('gpt2-medium').cuda()
