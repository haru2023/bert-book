# 8.固有表現抽出
# 実行方法："chap8"フォルダを作成し、"git clone --branch v2.0 https://github.com/stockmarkteam/ner-wikipedia-dataset"して、VSCodeでChapter8.pyを実行

# 8-1: カレントディレクトリを "./chap8/" にする
import os
file_directory = os.path.dirname(os.path.abspath(__file__)) # fileのあるディレクトリのパスを取得
target_directory = os.path.join(file_directory, 'chap8') # './chap8/'へのパスを構築
os.chdir(target_directory) # カレントディレクトリを変更
#// !mkdir chap8
#// %cd ./chap8

# 8-2: 必要なライブラリのインストール
#// !pip install transformers==4.18.0 fugashi==1.1.0 ipadic==1.0.0 pytorch-lightning==1.6.1

# 8-3: 固有表現認識のためのライブラリと設定をインポート
import itertools
import random
import json
from tqdm import tqdm
import numpy as np
import unicodedata

import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer, BertForTokenClassification
import pytorch_lightning as pl

# 日本語学習済みモデルの指定
MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'

# 8-4: テキストの正規化関数の定義とテスト
normalize = lambda s: unicodedata.normalize("NFKC",s) # NFKC形式での正規化を行う関数の定義
print(f'ＡＢＣ -> {normalize("ＡＢＣ")}' )  # 全角アルファベットを半角に変換して出力
print(f'ABC -> {normalize("ABC")}' )        # 半角アルファベットはそのまま出力
print(f'１２３ -> {normalize("１２３")}' )  # 全角数字を半角に変換して出力
print(f'123 -> {normalize("123")}' )        # 半角数字はそのまま出力
print(f'アイウ -> {normalize("アイウ")}' )  # 全角カタカナはそのまま出力
print(f'ｱｲｳ -> {normalize("ｱｲｳ")}' )        # 半角カタカナを全角に変換して出力

# 8-5
class NER_tokenizer(BertJapaneseTokenizer):
       
    def encode_plus_tagged(self, text, entities, max_length):
        """
        文章とそれに含まれる固有表現が与えられた時に、
        符号化とラベル列の作成を行う。
        """
        # 固有表現の前後でtextを分割し、それぞれのラベルをつけておく。
        entities = sorted(entities, key=lambda x: x['span'][0])
        splitted = [] # 分割後の文字列を追加していく
        position = 0
        for entity in entities:
            start = entity['span'][0]
            end = entity['span'][1]
            label = entity['type_id']
            # 固有表現ではないものには0のラベルを付与
            splitted.append({'text':text[position:start], 'label':0}) 
            # 固有表現には、固有表現のタイプに対応するIDをラベルとして付与
            splitted.append({'text':text[start:end], 'label':label}) 
            position = end
        splitted.append({'text': text[position:], 'label':0})
        splitted = [ s for s in splitted if s['text'] ] # 長さ0の文字列は除く

        # 分割されたそれぞれの文字列をトークン化し、ラベルをつける。
        tokens = [] # トークンを追加していく
        labels = [] # トークンのラベルを追加していく
        for text_splitted in splitted:
            text = text_splitted['text']
            label = text_splitted['label']
            tokens_splitted = self.tokenize(text)
            labels_splitted = [label] * len(tokens_splitted)
            tokens.extend(tokens_splitted)
            labels.extend(labels_splitted)

        # 符号化を行いBERTに入力できる形式にする。
        input_ids = self.convert_tokens_to_ids(tokens)
        encoding = self.prepare_for_model(
            input_ids, 
            max_length=max_length, 
            padding='max_length', 
            truncation=True
        ) # input_idsをencodingに変換
        # 特殊トークン[CLS]、[SEP]のラベルを0にする。
        labels = [0] + labels[:max_length-2] + [0] 
        # 特殊トークン[PAD]のラベルを0にする。
        labels = labels + [0]*( max_length - len(labels) ) 
        encoding['labels'] = labels

        return encoding

    def encode_plus_untagged(
        self, text, max_length=None, return_tensors=None
    ):
        """
        文章をトークン化し、それぞれのトークンの文章中の位置も特定しておく。
        """
        # 文章のトークン化を行い、
        # それぞれのトークンと文章中の文字列を対応づける。
        tokens = [] # トークンを追加していく。
        tokens_original = [] # トークンに対応する文章中の文字列を追加していく。
        words = self.word_tokenizer.tokenize(text) # MeCabで単語に分割
        for word in words:
            # 単語をサブワードに分割
            tokens_word = self.subword_tokenizer.tokenize(word) 
            tokens.extend(tokens_word)
            if tokens_word[0] == '[UNK]': # 未知語への対応
                tokens_original.append(word)
            else:
                tokens_original.extend([
                    token.replace('##','') for token in tokens_word
                ])

        # 各トークンの文章中での位置を調べる。（空白の位置を考慮する）
        position = 0
        spans = [] # トークンの位置を追加していく。
        for token in tokens_original:
            l = len(token)
            while 1:
                if token != text[position:position+l]:
                    position += 1
                else:
                    spans.append([position, position+l])
                    position += l
                    break

        # 符号化を行いBERTに入力できる形式にする。
        input_ids = self.convert_tokens_to_ids(tokens) 
        encoding = self.prepare_for_model(
            input_ids, 
            max_length=max_length, 
            padding='max_length' if max_length else False, 
            truncation=True if max_length else False
        )
        sequence_length = len(encoding['input_ids'])
        # 特殊トークン[CLS]に対するダミーのspanを追加。
        spans = [[-1, -1]] + spans[:sequence_length-2] 
        # 特殊トークン[SEP]、[PAD]に対するダミーのspanを追加。
        spans = spans + [[-1, -1]] * ( sequence_length - len(spans) ) 

        # 必要に応じてtorch.Tensorにする。
        if return_tensors == 'pt':
            encoding = { k: torch.tensor([v]) for k, v in encoding.items() }

        return encoding, spans

    def convert_bert_output_to_entities(self, text, labels, spans):
        """
        文章、ラベル列の予測値、各トークンの位置から固有表現を得る。
        """
        # labels, spansから特殊トークンに対応する部分を取り除く
        labels = [label for label, span in zip(labels, spans) if span[0] != -1]
        spans = [span for span in spans if span[0] != -1]

        # 同じラベルが連続するトークンをまとめて、固有表現を抽出する。
        entities = []
        for label, group \
            in itertools.groupby(enumerate(labels), key=lambda x: x[1]):
            
            group = list(group)
            start = spans[group[0][0]][0]
            end = spans[group[-1][0]][1]

            if label != 0: # ラベルが0以外ならば、新たな固有表現として追加。
                entity = {
                    "name": text[start:end],
                    "span": [start, end],
                    "type_id": label
                }
                entities.append(entity)

        return entities

# 8-6: トークナイザの初期化
tokenizer = NER_tokenizer.from_pretrained(MODEL_NAME) # 事前学習済みのモデルを使ってトークナイザを初期化

# 8-7: タグ付けされたテキストの符号化
text = '昨日のみらい事務所との打ち合わせは順調だった。'
entities = [
    {'name': 'みらい事務所', 'span': [3,9], 'type_id': 1} # 固有表現の指定
]

encoding = tokenizer.encode_plus_tagged(
    text, entities, max_length=20 # テキストを符号化し、固有表現にタグを付ける
)
print(encoding) # 符号化された結果を出力

# 8-8: タグなしテキストの符号化
text = '騰訊の英語名はTencent Holdings Ltdである。'
encoding, spans = tokenizer.encode_plus_untagged(
    text, return_tensors='pt' # テキストを符号化し、トークンの位置情報も取得
)
print('# encoding') # 符号化結果の出力
print(encoding)
print('# spans') # トークンの位置情報の出力
print(spans)

# 8-9: 予測されたラベルから固有表現を抽出
labels_predicted = [0,1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0] # 予測されたラベル
entities = tokenizer.convert_bert_output_to_entities(
    text, labels_predicted, spans # ラベルとトークン位置情報から固有表現を抽出
)
print(entities) # 抽出された固有表現の出力

# 8-10: BERTモデルの初期化とGPUへの転送
tokenizer = NER_tokenizer.from_pretrained(MODEL_NAME)
bert_tc = BertForTokenClassification.from_pretrained(
    MODEL_NAME, num_labels=4 # 固有表現のクラス数を指定
)
bert_tc = bert_tc.cuda() # BERTモデルをGPUに転送

# 8-11: 固有表現認識の実行
text = 'AさんはB大学に入学した。'

# 符号化を行い、各トークンの文章中での位置も特定
encoding, spans = tokenizer.encode_plus_untagged(
    text, return_tensors='pt'
) 
encoding = { k: v.cuda() for k, v in encoding.items() } # 符号化されたデータをGPUに転送

# BERTでトークン毎の分類スコアを出力し、スコアの最も高いラベルを予測値とする
with torch.no_grad():
    output = bert_tc(**encoding)
    scores = output.logits
    labels_predicted = scores[0].argmax(-1).cpu().numpy().tolist()

# ラベル列を固有表現に変換
entities = tokenizer.convert_bert_output_to_entities(
    text, labels_predicted, spans
)
print(entities) # 抽出された固有表現の出力

# 8-12
data = [
    {
        'text': 'AさんはB大学に入学した。',
        'entities': [
            {'name': 'A', 'span': [0, 1], 'type_id': 2},
            {'name': 'B大学', 'span': [4, 7], 'type_id': 1}
        ]
    },
    {
        'text': 'CDE株式会社は新製品「E」を販売する。',
        'entities': [
            {'name': 'CDE株式会社', 'span': [0, 7], 'type_id': 1},
            {'name': 'E', 'span': [12, 13], 'type_id': 3}
        ]
    }
]

# 各データを符号化し、データローダを作成する。
max_length=32
dataset_for_loader = []
for sample in data:
    text = sample['text']
    entities = sample['entities']
    encoding = tokenizer.encode_plus_tagged(
        text, entities, max_length=max_length
    )
    encoding = { k: torch.tensor(v) for k, v in encoding.items() }
    dataset_for_loader.append(encoding)
dataloader = DataLoader(dataset_for_loader, batch_size=len(data))

# ミニバッチを取り出し損失を得る。
for batch in dataloader:
    batch = { k: v.cuda() for k, v in batch.items() } # GPU
    output = bert_tc(**batch) # BERTへ入力
    loss = output.loss # 損失

# 8-13
#// !git clone --branch v2.0 https://github.com/stockmarkteam/ner-wikipedia-dataset 

# 8-14: データのロード
with open('ner-wikipedia-dataset/ner.json', encoding='utf-8') as f:  # JSONファイルを開く
    dataset = json.load(f)  # JSONファイルを読み込み、データセットを作成

# 固有表現のタイプとIDを対応付る辞書
type_id_dict = {
    "人名": 1,
    "法人名": 2,
    "政治的組織名": 3,
    "その他の組織名": 4,
    "地名": 5,
    "施設名": 6,
    "製品名": 7,
    "イベント名": 8
} # 固有表現の種類とそれに対応するIDを辞書で定義

# カテゴリーをラベルに変更、文字列の正規化する
for sample in dataset:
    sample['text'] = unicodedata.normalize('NFKC', sample['text'])  # テキストを正規化
    for e in sample["entities"]:
        e['type_id'] = type_id_dict[e['type']]  # カテゴリーを対応するIDに変換
        del e['type']  # 元のカテゴリー情報を削除

# データセットの分割
random.shuffle(dataset)  # データセットをシャッフル
n = len(dataset)  # データセットの総数
n_train = int(n*0.6)  # 訓練データの数
n_val = int(n*0.2)  # 検証データの数
dataset_train = dataset[:n_train]  # 訓練データを分割
dataset_val = dataset[n_train:n_train+n_val]  # 検証データを分割
dataset_test = dataset[n_train+n_val:]  # テストデータを分割

# 8-15: データセットの作成
def create_dataset(tokenizer, dataset, max_length):
    """
    データセットをデータローダに入力できる形に整形。
    """
    dataset_for_loader = []
    for sample in dataset:
        text = sample['text']
        entities = sample['entities']
        encoding = tokenizer.encode_plus_tagged(
            text, entities, max_length=max_length
        )  # テキストと固有表現情報を用いて符号化
        encoding = { k: torch.tensor(v) for k, v in encoding.items() }  # 符号化結果をTensorに変換
        dataset_for_loader.append(encoding)  # データセットに追加
    return dataset_for_loader

# トークナイザのロード
tokenizer = NER_tokenizer.from_pretrained(MODEL_NAME)  # 事前学習済みモデルを使用してトークナイザをロード

# データセットの作成
max_length = 128  # トークンの最大長を設定
dataset_train_for_loader = create_dataset(
    tokenizer, dataset_train, max_length
)  # 訓練データセットを作成
dataset_val_for_loader = create_dataset(
    tokenizer, dataset_val, max_length
)  # 検証データセットを作成

# データローダの作成
dataloader_train = DataLoader(
    dataset_train_for_loader, batch_size=32, shuffle=True
)  # 訓練データ用のデータローダを作成
dataloader_val = DataLoader(dataset_val_for_loader, batch_size=256)  # 検証データ用のデータローダを作成

# 8-16: PyTorch Lightningのモデル定義
class BertForTokenClassification_pl(pl.LightningModule):
        
    def __init__(self, model_name, num_labels, lr):
        super().__init__()
        self.save_hyperparameters()
        self.bert_tc = BertForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels  # 固有表現のクラス数を指定
        )
        
    def training_step(self, batch, batch_idx):
        output = self.bert_tc(**batch)
        loss = output.loss  # 損失を計算
        self.log('train_loss', loss)  # 訓練損失を記録
        return loss
        
    def validation_step(self, batch, batch_idx):
        output = self.bert_tc(**batch)
        val_loss = output.loss  # 検証損失を計算
        self.log('val_loss', val_loss)  # 検証損失を記録
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)  # 最適化アルゴリズムを設定

checkpoint = pl.callbacks.ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_top_k=1,
    save_weights_only=True,
    dirpath='model/'
)  # モデルチェックポイントの設定

trainer = pl.Trainer(
    gpus=1,
    max_epochs=5,
    callbacks=[checkpoint]
)  # トレーナーの設定

# ファインチューニング
model = BertForTokenClassification_pl(
    MODEL_NAME, num_labels=9, lr=1e-5
)  # モデルのインスタンスを作成
trainer.fit(model, dataloader_train, dataloader_val)  # モデルの訓練
best_model_path = checkpoint.best_model_path  # 最適なモデルのパスを取得

# 8-17: BERTモデルを使用した固有表現抽出関数
def predict(text, tokenizer, bert_tc):
    """
    BERTで固有表現抽出を行うための関数。
    """
    # 符号化
    encoding, spans = tokenizer.encode_plus_untagged(
        text, return_tensors='pt'  # テキストを符号化し、トークンの位置情報も取得
    )
    encoding = { k: v.cuda() for k, v in encoding.items() }  # GPUにデータを転送

    # ラベルの予測値の計算
    with torch.no_grad():
        output = bert_tc(**encoding)
        scores = output.logits
        labels_predicted = scores[0].argmax(-1).cpu().numpy().tolist()  # 最も確率の高いラベルを予測値として選択

    # ラベル列を固有表現に変換
    entities = tokenizer.convert_bert_output_to_entities(
        text, labels_predicted, spans  # ラベルとトークン位置情報から固有表現を抽出
    )

    return entities

# トークナイザのロード
tokenizer = NER_tokenizer.from_pretrained(MODEL_NAME)  # 事前学習済みモデルを使用してトークナイザをロード

# ファインチューニングしたモデルをロードし、GPUに載せる
model = BertForTokenClassification_pl.load_from_checkpoint(
    best_model_path  # 保存されたベストモデルのパス
)
bert_tc = model.bert_tc.cuda()  # モデルをGPUに転送

# 固有表現抽出
# 注：以下ではコードのわかりやすさのために、1データづつ処理しているが、
# バッチ化して処理を行った方が処理時間は短い
entities_list = []  # 正解の固有表現を追加
entities_predicted_list = []  # 抽出された固有表現を追加
for sample in tqdm(dataset_test):
    text = sample['text']
    entities_predicted = predict(text, tokenizer, bert_tc)  # BERTで固有表現を予測
    entities_list.append(sample['entities'])
    entities_predicted_list.append(entities_predicted)

# 8-18: 固有表現抽出の結果表示
print("# 正解")
print(entities_list[0])  # テストデータの1番目の正解固有表現を表示
print("# 抽出")
print(entities_predicted_list[0])  # 抽出された固有表現を表示

# 8-19: モデルの評価関数
def evaluate_model(entities_list, entities_predicted_list, type_id=None):
    """
    正解と予測を比較し、モデルの固有表現抽出の性能を評価する。
    type_idがNoneのときは、全ての固有表現のタイプに対して評価する。
    type_idが整数を指定すると、その固有表現のタイプのIDに対して評価を行う。
    """
    num_entities = 0  # 固有表現(正解)の個数
    num_predictions = 0  # BERTにより予測された固有表現の個数
    num_correct = 0  # BERTにより予測のうち正解であった固有表現の数

    # それぞれの文章で予測と正解を比較
    # 予測は文章中の位置とタイプIDが一致すれば正解とみなす。
    for entities, entities_predicted in zip(entities_list, entities_predicted_list):
        if type_id:
            entities = [e for e in entities if e['type_id'] == type_id]
            entities_predicted = [e for e in entities_predicted if e['type_id'] == type_id]

        get_span_type = lambda e: (e['span'][0], e['span'][1], e['type_id'])
        set_entities = set(get_span_type(e) for e in entities)
        set_entities_predicted = set(get_span_type(e) for e in entities_predicted)

        num_entities += len(entities)
        num_predictions += len(entities_predicted)
        num_correct += len(set_entities & set_entities_predicted)

    # 性能指標の計算
    precision = num_correct / num_predictions  # 適合率
    recall = num_correct / num_entities  # 再現率
    f_value = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0  # F値

    result = {
        'num_entities': num_entities,
        'num_predictions': num_predictions,
        'num_correct': num_correct,
        'precision': precision,
        'recall': recall,
        'f_value': f_value
    }

    return result

# 8-20: モデル評価の実行と結果表示
print(evaluate_model(entities_list, entities_predicted_list))  # 全データに対する評価結果を表示

# 8-21: BIO方式を用いた固有表現認識用トークナイザークラス
class NER_tokenizer_BIO(BertJapaneseTokenizer):

    # 初期化時に固有表現のカテゴリーの数`num_entity_type`を受け入れるようにする。
    def __init__(self, *args, **kwargs):
        self.num_entity_type = kwargs.pop('num_entity_type')  # 固有表現のタイプ数を取得
        super().__init__(*args, **kwargs)

    def encode_plus_tagged(self, text, entities, max_length):
        """
        文章とそれに含まれる固有表現が与えられた時に、
        符号化とラベル列の作成を行う。
        """
        # 固有表現の前後でテキストを分割し、ラベルを付与
        splitted = []  # 分割後の文字列を追加
        position = 0
        for entity in entities:
            start = entity['span'][0]
            end = entity['span'][1]
            label = entity['type_id']
            splitted.append({'text': text[position:start], 'label': 0})
            splitted.append({'text': text[start:end], 'label': label})
            position = end
        splitted.append({'text': text[position:], 'label': 0})
        splitted = [s for s in splitted if s['text']]

        # 分割されたテキストをトークン化し、ラベルを付与
        tokens = []  # トークンを追加
        labels = []  # ラベルを追加
        for s in splitted:
            tokens_splitted = self.tokenize(s['text'])
            label = s['label']
            if label > 0:  # 固有表現の場合
                # まずトークン全てにI-タグを付与
                labels_splitted = [label + self.num_entity_type] * len(tokens_splitted)  # I-タグ
                # 先頭のトークンをB-タグにする
                labels_splitted[0] = label
            else:  # 固有表現以外の場合
                labels_splitted = [0] * len(tokens_splitted)
            
            tokens.extend(tokens_splitted)
            labels.extend(labels_splitted)

        # BERTに入力できる形式に符号化
        input_ids = self.convert_tokens_to_ids(tokens)
        encoding = self.prepare_for_model(
            input_ids, 
            max_length=max_length, 
            padding='max_length',
            truncation=True
        ) 

        # 特殊トークンに対するラベルを追加
        labels = [0] + labels[:max_length - 2] + [0]
        labels = labels + [0] * (max_length - len(labels))
        encoding['labels'] = labels

        return encoding

    def encode_plus_untagged(
        self, text, max_length=None, return_tensors=None
    ):
        """
        文章をトークン化し、それぞれのトークンの文章中の位置も特定しておく。
        IO法のトークナイザのencode_plus_untaggedと同じ
        """
        # 文章のトークン化を行い、
        # それぞれのトークンと文章中の文字列を対応づける。
        tokens = [] # トークンを追加していく。
        tokens_original = [] # トークンに対応する文章中の文字列を追加していく。
        words = self.word_tokenizer.tokenize(text) # MeCabで単語に分割
        for word in words:
            # 単語をサブワードに分割
            tokens_word = self.subword_tokenizer.tokenize(word) 
            tokens.extend(tokens_word)
            if tokens_word[0] == '[UNK]': # 未知語への対応
                tokens_original.append(word)
            else:
                tokens_original.extend([
                    token.replace('##','') for token in tokens_word
                ])

        # 各トークンの文章中での位置を調べる。（空白の位置を考慮する）
        position = 0
        spans = [] # トークンの位置を追加していく。
        for token in tokens_original:
            l = len(token)
            while 1:
                if token != text[position:position+l]:
                    position += 1
                else:
                    spans.append([position, position+l])
                    position += l
                    break

        # 符号化を行いBERTに入力できる形式にする。
        input_ids = self.convert_tokens_to_ids(tokens) 
        encoding = self.prepare_for_model(
            input_ids, 
            max_length=max_length, 
            padding='max_length' if max_length else False, 
            truncation=True if max_length else False
        )
        sequence_length = len(encoding['input_ids'])
        # 特殊トークン[CLS]に対するダミーのspanを追加。
        spans = [[-1, -1]] + spans[:sequence_length-2] 
        # 特殊トークン[SEP]、[PAD]に対するダミーのspanを追加。
        spans = spans + [[-1, -1]] * ( sequence_length - len(spans) ) 

        # 必要に応じてtorch.Tensorにする。
        if return_tensors == 'pt':
            encoding = { k: torch.tensor([v]) for k, v in encoding.items() }

        return encoding, spans

    @staticmethod
    def Viterbi(scores_bert, num_entity_type, penalty=10000):
        """
        Viterbiアルゴリズムで最適解を求める。
        """
        m = 2 * num_entity_type + 1  # タグの総数 (B, Iタグそれぞれに対してnum_entity_type種類とOタグ)
        penalty_matrix = np.zeros([m, m])  # 罰点行列の初期化
        for i in range(m):
            for j in range(1 + num_entity_type, m):
                if not ((i == j) or (i + num_entity_type == j)):
                    penalty_matrix[i, j] = penalty  # 不適切なタグの遷移には罰点を与える
        
        path = [[i] for i in range(m)]  # 各タグから始まるパスを初期化
        scores_path = scores_bert[0] - penalty_matrix[0, :]  # 最初のトークンに対するスコア
        scores_bert = scores_bert[1:]  # 最初のトークンを除いた残りのスコア

        # 以降のトークンに対してViterbiアルゴリズムを適用
        for scores in scores_bert:
            assert len(scores) == m  # スコアの長さのチェック
            score_matrix = np.array(scores_path).reshape(-1, 1) + np.array(scores).reshape(1, -1) - penalty_matrix
            scores_path = score_matrix.max(axis=0)  # 各タグに到達する最大スコア
            argmax = score_matrix.argmax(axis=0)  # 最大スコアを与えるパスのインデックス
            path_new = []
            for i, idx in enumerate(argmax):
                path_new.append(path[idx] + [i])  # 新しいパスを更新
            path = path_new

        labels_optimal = path[np.argmax(scores_path)]  # 全トークンに対する最適なタグのシーケンスを選択
        return labels_optimal

    def convert_bert_output_to_entities(self, text, scores, spans):
        """
        文章、分類スコア、各トークンの位置から固有表現を得る。
        分類スコアはサイズが（系列長、ラベル数）の2次元配列
        """
        assert len(spans) == len(scores)
        num_entity_type = self.num_entity_type
        
        # 特殊トークンに対応する部分を取り除く
        scores = [score for score, span in zip(scores, spans) if span[0]!=-1]
        spans = [span for span in spans if span[0]!=-1]

        # Viterbiアルゴリズムでラベルの予測値を決める。
        labels = self.Viterbi(scores, num_entity_type)

        # 同じラベルが連続するトークンをまとめて、固有表現を抽出する。
        entities = []
        for label, group \
            in itertools.groupby(enumerate(labels), key=lambda x: x[1]):
            
            group = list(group)
            start = spans[group[0][0]][0]
            end = spans[group[-1][0]][1]

            if label != 0: # 固有表現であれば
                if 1 <= label <= num_entity_type:
                     # ラベルが`B-`ならば、新しいentityを追加
                    entity = {
                        "name": text[start:end],
                        "span": [start, end],
                        "type_id": label
                    }
                    entities.append(entity)
                else:
                    # ラベルが`I-`ならば、直近のentityを更新
                    entity['span'][1] = end 
                    entity['name'] = text[entity['span'][0]:entity['span'][1]]
                
        return entities

# 8-22: トークナイザのロード
# 固有表現のカテゴリーの数`num_entity_type`を入力に入れる必要がある。
tokenizer = NER_tokenizer_BIO.from_pretrained(
    MODEL_NAME,
    num_entity_type=8  # 固有表現のカテゴリー数を指定
)

# データセットの作成
max_length = 128  # 最大トークン長を指定
dataset_train_for_loader = create_dataset(
    tokenizer, dataset_train, max_length  # 訓練データセットを作成
)
dataset_val_for_loader = create_dataset(
    tokenizer, dataset_val, max_length  # 検証データセットを作成
)

# データローダの作成
dataloader_train = DataLoader(
    dataset_train_for_loader, batch_size=32, shuffle=True  # 訓練データ用のデータローダ
)
dataloader_val = DataLoader(dataset_val_for_loader, batch_size=256)  # 検証データ用のデータローダ

# 8-23: ファインチューニングと性能評価
checkpoint = pl.callbacks.ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_top_k=1,
    save_weights_only=True,
    dirpath='model_BIO/'  # モデル保存先のディレクトリを指定
)

trainer = pl.Trainer(
    gpus=1,
    max_epochs=5,
    callbacks=[checkpoint]  # トレーナーの設定
)

# PyTorch Lightningのモデルのロード
num_entity_type = 8  # 固有表現のタイプ数
num_labels = 2*num_entity_type + 1  # ラベル数を計算（B, I, Oタグ分）
model = BertForTokenClassification_pl(
    MODEL_NAME, num_labels=num_labels, lr=1e-5  # モデルを初期化
)

# ファインチューニングを実行
trainer.fit(model, dataloader_train, dataloader_val)
best_model_path = checkpoint.best_model_path  # 最良モデルのパスを取得

# 性能評価
model = BertForTokenClassification_pl.load_from_checkpoint(
    best_model_path  # 保存された最良モデルをロード
) 
bert_tc = model.bert_tc.cuda()  # モデルをGPUに転送

entities_list = []  # 正解の固有表現リスト
entities_predicted_list = []  # 抽出された固有表現リスト
for sample in tqdm(dataset_test):  # テストデータに対して予測
    text = sample['text']
    encoding, spans = tokenizer.encode_plus_untagged(
        text, return_tensors='pt'  # テキストを符号化
    )
    encoding = { k: v.cuda() for k, v in encoding.items() }  # データをGPUに転送
    
    with torch.no_grad():
        output = bert_tc(**encoding)
        scores = output.logits
        scores = scores[0].cpu().numpy().tolist()  # スコアを取得
        
    # 分類スコアから固有表現を抽出
    entities_predicted = tokenizer.convert_bert_output_to_entities(
        text, scores, spans
    )

    entities_list.append(sample['entities'])  # 正解の固有表現を追加
    entities_predicted_list.append(entities_predicted)  # 抽出された固有表現を追加

# モデルの性能を評価して表示
print(evaluate_model(entities_list, entities_predicted_list))
