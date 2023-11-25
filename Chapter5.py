# 5.BERT
# 実行方法：VSCodeでChapter5.pyを実行

# 5-1: 必要なライブラリをインポート
#// !pip install transformers==4.18.0 fugashi==1.1.0 ipadic==1.0.0  # transformers, fugashi, ipadicをインストール

# 5-2: ライブラリをインポート
import numpy as np  # NumPyをインポート
import torch  # PyTorchをインポート
from transformers import BertJapaneseTokenizer, BertForMaskedLM  # transformersからBERTの日本語トークナイザーとMaskedLMモデルをインポート

# 5-3: BERTモデルの準備
model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'  # 使用するBERTモデルの名前を指定
tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)  # 指定したモデル名でトークナイザーを初期化
bert_mlm = BertForMaskedLM.from_pretrained(model_name)  # 指定したモデル名でMaskedLMモデルを初期化
bert_mlm = bert_mlm.cuda()  # モデルをGPUに転送

# 5-4: テキストのトークン化
text = '今日は[MASK]へ行く。'  # トークン化するテキスト
tokens = tokenizer.tokenize(text)  # テキストをトークン化
print(tokens)  # トークン化された結果を表示

# 5-5: 文章をBERTモデルに入力し、スコアを得る
input_ids = tokenizer.encode(text, return_tensors='pt')  # テキストを符号化
input_ids = input_ids.cuda()  # 符号化されたテキストをGPUに転送

# BERTに入力し、分類スコアを得る。
# 系列長を揃える必要がないので、単にiput_idsのみを入力します。
with torch.no_grad():  # 勾配計算を無効にして推論モード
    output = bert_mlm(input_ids=input_ids)  # BERTモデルに入力
    scores = output.logits  # 出力からスコアを取得

# 5-6: [MASK]を最も可能性の高いトークンで置き換える
# ID列で'[MASK]'(IDは4)の位置を調べる
mask_position = input_ids[0].tolist().index(4)  # [MASK]トークンの位置を特定

# スコアが最も良いトークンのIDを取り出し、トークンに変換する。
id_best = scores[0, mask_position].argmax(-1).item()  # 最もスコアが高いトークンのIDを取得
token_best = tokenizer.convert_ids_to_tokens(id_best)  # トークンIDをトークンに変換
token_best = token_best.replace('##', '')  # 不要な文字を削除

# [MASK]を上で求めたトークンで置き換える。
text = text.replace('[MASK]', token_best)  # [MASK]を置き換えたテキストを作成
print(text)  # 置き換えたテキストを表示

# 5-7: スコアの上位トークンで[MASK]を置き換える関数
def predict_mask_topk(text, tokenizer, bert_mlm, num_topk):
    """
    文章中の最初の[MASK]をスコアの上位のトークンに置き換える。
    上位何位まで使うかは、num_topkで指定。
    出力は穴埋めされた文章のリストと、置き換えられたトークンのスコアのリスト。
    """
    input_ids = tokenizer.encode(text, return_tensors='pt')  # テキストを符号化
    input_ids = input_ids.cuda()  # 符号化されたテキストをGPUに転送
    with torch.no_grad():
        output = bert_mlm(input_ids=input_ids)  # BERTモデルに入力
    scores = output.logits  # 出力からスコアを取得

    # スコアが上位のトークンとスコアを求める。
    mask_position = input_ids[0].tolist().index(4)  # [MASK]トークンの位置を特定
    topk = scores[0, mask_position].topk(num_topk)  # スコアが上位のトークンを取得
    ids_topk = topk.indices  # トークンのID
    tokens_topk = tokenizer.convert_ids_to_tokens(ids_topk)  # IDをトークンに変換
    scores_topk = topk.values.cpu().numpy()  # スコアを取得

    # 文章中の[MASK]を上で求めたトークンで置き換える。
    text_topk = []  # 置き換えたテキストのリスト
    for token in tokens_topk:
        token = token.replace('##', '')  # 不要な文字を削除
        text_topk.append(text.replace('[MASK]', token, 1))  # [MASK]を置き換え

    return text_topk, scores_topk

text = '今日は[MASK]へ行く。'
text_topk, _ = predict_mask_topk(text, tokenizer, bert_mlm, 10)
print(*text_topk, sep='\n')  # 結果を表示

# 5-8: 貪欲法による[MASK]の穴埋め
def greedy_prediction(text, tokenizer, bert_mlm):
    """
    [MASK]を含む文章を入力として、貪欲法で穴埋めを行った文章を出力する。
    """
    # 前から順に[MASK]を一つづつ、スコアの最も高いトークンに置き換える。
    for _ in range(text.count('[MASK]')):
        text = predict_mask_topk(text, tokenizer, bert_mlm, 1)[0][0]  # スコアが最も高いトークンで[MASK]を置き換え
    return text

text = '今日は[MASK][MASK]へ行く。'
print(greedy_prediction(text, tokenizer, bert_mlm))  # 穴埋めしたテキストを表示

# 5-9: 複数の[MASK]を含むテキストの穴埋め
text = '今日は[MASK][MASK][MASK][MASK][MASK]'
print(greedy_prediction(text, tokenizer, bert_mlm))  # 穴埋めしたテキストを表示

# 5-10: ビームサーチによる穴埋め
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
            text_topk_inner, scores_topk_inner = predict_mask_topk(text_mask, tokenizer, bert_mlm, num_topk)
            text_candidates.extend(text_topk_inner)
            score_candidates.append(score + scores_topk_inner)

        # 穴埋めにより生成された文章の中から合計スコアの高いものを選ぶ。
        score_candidates = np.hstack(score_candidates)
        idx_list = score_candidates.argsort()[::-1][:num_topk]
        text_topk = [text_candidates[idx] for idx in idx_list]
        scores_topk = score_candidates[idx_list]

    return text_topk

text = "今日は[MASK][MASK]へ行く。"
text_topk = beam_search(text, tokenizer, bert_mlm, 10)
print(*text_topk, sep='\n')

# 5-11: 複数の[MASK]を含むテキストの穴埋め（ビームサーチ）
text = '今日は[MASK][MASK][MASK][MASK][MASK]'
text_topk = beam_search(text, tokenizer, bert_mlm, 10)
print(*text_topk, sep='\n')
