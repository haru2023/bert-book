# 4.Transformer

# 4-1
#// !pip install transformers==4.18.0 fugashi==1.1.0 ipadic==1.0.0

# 4-2
import torch
from transformers import BertJapaneseTokenizer, BertModel

# 4-3
model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)

# 4-4
tokenizer.tokenize('明日は自然言語処理の勉強をしよう。')

# 4-5
tokenizer.tokenize('明日はマシンラーニングの勉強をしよう。')

# 4-6
tokenizer.tokenize('機械学習を中国語にすると机器学习だ。')

# 4-7
input_ids = tokenizer.encode('明日は自然言語処理の勉強をしよう。')
print(input_ids)

# 4-8
tokenizer.convert_ids_to_tokens(input_ids)

# 4-9
text = '明日の天気は晴れだ。'
encoding = tokenizer(
    text, max_length=12, padding='max_length', truncation=True
)
print('# encoding:')
print(encoding)

tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
print('# tokens:')
print(tokens)

# 4-10
encoding = tokenizer(
    text, max_length=6, padding='max_length', truncation=True
)
tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
print(tokens)

# 4-11
text_list = ['明日の天気は晴れだ。','パソコンが急に動かなくなった。']
tokenizer(
    text_list, max_length=10, padding='max_length', truncation=True
)

# 4-12
tokenizer(text_list, padding='longest')

# 4-13
tokenizer(
    text_list,
    max_length=10,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

# 4-14
# モデルのロード
model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
bert = BertModel.from_pretrained(model_name)

# BERTをGPUに載せる
bert = bert.cuda() 

# 4-15
print(bert.config)

# 4-16
text_list = [
    '明日は自然言語処理の勉強をしよう。',
    '明日はマシーンラーニングの勉強をしよう。'
]

# 文章の符号化
encoding = tokenizer(
    text_list,
    max_length=32,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

# データをGPUに載せる
encoding = { k: v.cuda() for k, v in encoding.items() } 

# BERTでの処理
output = bert(**encoding) # それぞれの入力は2次元のtorch.Tensor
last_hidden_state = output.last_hidden_state # 最終層の出力

# 4-17
output = bert(
    input_ids=encoding['input_ids'], 
    attention_mask=encoding['attention_mask'],
    token_type_ids=encoding['token_type_ids']
)

# 4-18
print(last_hidden_state.size()) #テンソルのサイズ

# 4-19
with torch.no_grad():
    output = bert(**encoding)
    last_hidden_state = output.last_hidden_state

# 4-20
last_hidden_state = last_hidden_state.cpu() # CPUにうつす。
last_hidden_state = last_hidden_state.numpy() # numpy.ndarrayに変換
last_hidden_state = last_hidden_state.tolist() # リストに変換
