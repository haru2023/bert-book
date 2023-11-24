# 5.BERT

# 5-1
#// !pip install transformers==4.18.0 fugashi==1.1.0 ipadic==1.0.0

# 5-2
import numpy as np
import torch
from transformers import BertJapaneseTokenizer, BertForMaskedLM

# 5-3
model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
bert_mlm = BertForMaskedLM.from_pretrained(model_name)
bert_mlm = bert_mlm.cuda()

# 5-4
text = '今日は[MASK]へ行く。'
tokens = tokenizer.tokenize(text)
print(tokens)

# 5-5
# 文章を符号化し、GPUに配置する。
input_ids = tokenizer.encode(text, return_tensors='pt')
input_ids = input_ids.cuda()

# BERTに入力し、分類スコアを得る。
# 系列長を揃える必要がないので、単にiput_idsのみを入力します。
with torch.no_grad():
    output = bert_mlm(input_ids=input_ids)
    scores = output.logits

# 5-6
# ID列で'[MASK]'(IDは4)の位置を調べる
mask_position = input_ids[0].tolist().index(4) 

# スコアが最も良いトークンのIDを取り出し、トークンに変換する。
id_best = scores[0, mask_position].argmax(-1).item()
token_best = tokenizer.convert_ids_to_tokens(id_best)
token_best = token_best.replace('##', '')

# [MASK]を上で求めたトークンで置き換える。
text = text.replace('[MASK]',token_best)

print(text)

# 5-7
def predict_mask_topk(text, tokenizer, bert_mlm, num_topk):
    """
    文章中の最初の[MASK]をスコアの上位のトークンに置き換える。
    上位何位まで使うかは、num_topkで指定。
    出力は穴埋めされた文章のリストと、置き換えられたトークンのスコアのリスト。
    """
    # 文章を符号化し、BERTで分類スコアを得る。
    input_ids = tokenizer.encode(text, return_tensors='pt')
    input_ids = input_ids.cuda()
    with torch.no_grad():
        output = bert_mlm(input_ids=input_ids)
    scores = output.logits

    # スコアが上位のトークンとスコアを求める。
    mask_position = input_ids[0].tolist().index(4) 
    topk = scores[0, mask_position].topk(num_topk)
    ids_topk = topk.indices # トークンのID
    tokens_topk = tokenizer.convert_ids_to_tokens(ids_topk) # トークン
    scores_topk = topk.values.cpu().numpy() # スコア

    # 文章中の[MASK]を上で求めたトークンで置き換える。
    text_topk = [] # 穴埋めされたテキストを追加する。
    for token in tokens_topk:
        token = token.replace('##', '')
        text_topk.append(text.replace('[MASK]', token, 1))

    return text_topk, scores_topk

text = '今日は[MASK]へ行く。'
text_topk, _ = predict_mask_topk(text, tokenizer, bert_mlm, 10)
print(*text_topk, sep='\n')

# 5-8
def greedy_prediction(text, tokenizer, bert_mlm):
    """
    [MASK]を含む文章を入力として、貪欲法で穴埋めを行った文章を出力する。
    """
    # 前から順に[MASK]を一つづつ、スコアの最も高いトークンに置き換える。
    for _ in range(text.count('[MASK]')):
        text = predict_mask_topk(text, tokenizer, bert_mlm, 1)[0][0]
    return text

text = '今日は[MASK][MASK]へ行く。'
greedy_prediction(text, tokenizer, bert_mlm)

# 5-9
text = '今日は[MASK][MASK][MASK][MASK][MASK]'
greedy_prediction(text, tokenizer, bert_mlm)

# 5-10
def beam_search(text, tokenizer, bert_mlm, num_topk):
    """
    ビームサーチで文章の穴埋めを行う。
    """
    num_mask = text.count('[MASK]')
    text_topk = [text]
    scores_topk = np.array([0])
    for _ in range(num_mask):
        # 現在得られている、それぞれの文章に対して、
        # 最初の[MASK]をスコアが上位のトークンで穴埋めする。
        text_candidates = [] # それぞれの文章を穴埋めした結果を追加する。
        score_candidates = [] # 穴埋めに使ったトークンのスコアを追加する。
        for text_mask, score in zip(text_topk, scores_topk):
            text_topk_inner, scores_topk_inner = predict_mask_topk(
                text_mask, tokenizer, bert_mlm, num_topk
            )
            text_candidates.extend(text_topk_inner)
            score_candidates.append( score + scores_topk_inner )

        # 穴埋めにより生成された文章の中から合計スコアの高いものを選ぶ。
        score_candidates = np.hstack(score_candidates)
        idx_list = score_candidates.argsort()[::-1][:num_topk]
        text_topk = [ text_candidates[idx] for idx in idx_list ]
        scores_topk = score_candidates[idx_list]

    return text_topk

text = "今日は[MASK][MASK]へ行く。"
text_topk = beam_search(text, tokenizer, bert_mlm, 10)
print(*text_topk, sep='\n')

# 5-11
text = '今日は[MASK][MASK][MASK][MASK][MASK]'
text_topk = beam_search(text, tokenizer, bert_mlm, 10)
print(*text_topk, sep='\n')


