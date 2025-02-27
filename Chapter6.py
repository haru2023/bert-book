# 6.BERTによるテキスト分類
# 実行方法："chap6"フォルダを作成し、"https://www.rondhuit.com/download/ldcc-20140209.tar.gz"を解凍して、VSCodeでChapter6.pyを実行

# 6-1: カレントディレクトリを "./chap6/" にする
import os
file_directory = os.path.dirname(os.path.abspath(__file__)) # fileのあるディレクトリのパスを取得
target_directory = os.path.join(file_directory, 'chap6') # './chap6/'へのパスを構築
os.chdir(target_directory) # カレントディレクトリを変更

#// !mkdir chap6
#// %cd ./chap6

# 6-2: 必要なライブラリをインストール
#// !pip install transformers==4.18.0 fugashi==1.1.0 ipadic==1.0.0 pytorch-lightning==1.6.1

# 6-3: BERTに関連するライブラリのインポート
import random
import glob
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer, BertForSequenceClassification
import pytorch_lightning as pl

# 日本語の事前学習モデル
MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'

# 6-4: トークナイザとモデルの初期化
tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)
bert_sc = BertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
)
bert_sc = bert_sc.cuda() # GPUでの計算を可能にする

# 6-5: テキストデータとラベルの定義
text_list = [
    "この映画は面白かった。",
    "この映画の最後にはがっかりさせられた。",
    "この映画を見て幸せな気持ちになった。"
]
label_list = [1,0,1]

# データの符号化
encoding = tokenizer(
    text_list, 
    padding = 'longest',
    return_tensors='pt'
)
encoding = { k: v.cuda() for k, v in encoding.items() } # GPU用にデータを移動
labels = torch.tensor(label_list).cuda() # GPU用にラベルを移動

# 推論
with torch.no_grad():
    output = bert_sc.forward(**encoding)
scores = output.logits # 分類スコア
labels_predicted = scores.argmax(-1) # スコアが最も高いラベルを予測
num_correct = (labels_predicted==labels).sum().item() # 正解数
accuracy = num_correct/labels.size(0) # 精度

print("# scores:")
print(scores.size())
print("# predicted labels:")
print(labels_predicted)
print("# accuracy:")
print(accuracy)

# 6-6: 損失計算の例
encoding = tokenizer(
    text_list, 
    padding='longest',
    return_tensors='pt'
) 
encoding['labels'] = torch.tensor(label_list) # 入力にラベルを加える。
encoding = { k: v.cuda() for k, v in encoding.items() } # GPU用にデータを移動

# ロスの計算
output = bert_sc(**encoding)
loss = output.loss # 損失の取得
print(loss)

# 6-7: データのダウンロードと解凍
#// #データのダウンロード
#// !wget https://www.rondhuit.com/download/ldcc-20140209.tar.gz 
#// #ファイルの解凍
#// !tar -zxf ldcc-20140209.tar.gz 

# 6-8: ファイルの内容表示
#// !cat ./text/it-life-hack/it-life-hack-6342280.txt # ファイルを表示

# 6-9: データローダーの作成
dataset_for_loader = [
    {'data':torch.tensor([0,1]), 'labels':torch.tensor(0)},
    {'data':torch.tensor([2,3]), 'labels':torch.tensor(1)},
    {'data':torch.tensor([4,5]), 'labels':torch.tensor(2)},
    {'data':torch.tensor([6,7]), 'labels':torch.tensor(3)},
]
loader = DataLoader(dataset_for_loader, batch_size=2)

# データセットからミニバッチを取り出す
for idx, batch in enumerate(loader):
    print(f'# batch {idx}')
    print(batch)
    ## ファインチューニングではここでミニバッチ毎の処理を行う

# 6-10: シャッフルされたデータローダー
loader = DataLoader(dataset_for_loader, batch_size=2, shuffle=True)

for idx, batch in enumerate(loader):
    print(f'# batch {idx}')
    print(batch)

# 6-11: データセットの前処理
# カテゴリーのリスト
category_list = [
    'dokujo-tsushin',
    'it-life-hack',
    'kaden-channel',
    'livedoor-homme',
    'movie-enter',
    'peachy',
    'smax',
    'sports-watch',
    'topic-news'
]

# トークナイザのロード
tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)

# 各データの形式を整える
max_length = 128
dataset_for_loader = []
for label, category in enumerate(tqdm(category_list)):
    for file in glob.glob(f'./text/{category}/{category}*'):
        with open(file, encoding='utf-8') as f:
            lines = f.read().splitlines()
        text = '\n'.join(lines[3:]) # ファイルの4行目からを抜き出す。
        encoding = tokenizer(
            text,
            max_length=max_length, 
            padding='max_length',
            truncation=True
        )
        encoding['labels'] = label # ラベルを追加
        encoding = { k: torch.tensor(v) for k, v in encoding.items() }
        dataset_for_loader.append(encoding)

# 6-12: データセットの内容表示
print(dataset_for_loader[0])

# 6-13: データセットの分割
random.shuffle(dataset_for_loader) # ランダムにシャッフル
n = len(dataset_for_loader)
n_train = int(0.6*n)
n_val = int(0.2*n)
dataset_train = dataset_for_loader[:n_train] # 学習データ
dataset_val = dataset_for_loader[n_train:n_train+n_val] # 検証データ
dataset_test = dataset_for_loader[n_train+n_val:] # テストデータ

# データセットからデータローダを作成
# 学習データはshuffle=Trueにする。
dataloader_train = DataLoader(
    dataset_train, batch_size=32, shuffle=True
) 
dataloader_val = DataLoader(dataset_val, batch_size=256)
dataloader_test = DataLoader(dataset_test, batch_size=256)

# 6-14: PyTorch Lightning用のBERTモデルクラス
class BertForSequenceClassification_pl(pl.LightningModule):
        
    def __init__(self, model_name, num_labels, lr):
        # model_name: Transformersのモデルの名前
        # num_labels: ラベルの数
        # lr: 学習率

        super().__init__()

        # 引数のnum_labelsとlrを保存。
        # 例えば、self.hparams.lrでlrにアクセスできる。
        # チェックポイント作成時にも自動で保存される。
        self.save_hyperparameters() 

        # BERTのロード
        self.bert_sc = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )

    # 学習データのミニバッチ(`batch`)が与えられた時に損失を出力する関数を書く。
    # batch_idxはミニバッチの番号であるが今回は使わない。
    def training_step(self, batch, batch_idx):
        output = self.bert_sc(**batch)
        loss = output.loss
        self.log('train_loss', loss) # 損失を'train_loss'の名前でログをとる。
        return loss

    # 検証データのミニバッチが与えられた時に、
    # 検証データを評価する指標を計算する関数を書く。
    def validation_step(self, batch, batch_idx):
        output = self.bert_sc(**batch)
        val_loss = output.loss
        self.log('val_loss', val_loss) # 損失を'val_loss'の名前でログをとる。

    # テストデータのミニバッチが与えられた時に、
    # テストデータを評価する指標を計算する関数を書く。
    def test_step(self, batch, batch_idx):
        labels = batch.pop('labels') # バッチからラベルを取得
        output = self.bert_sc(**batch)
        labels_predicted = output.logits.argmax(-1)
        num_correct = ( labels_predicted == labels ).sum().item()
        accuracy = num_correct/labels.size(0) #精度
        self.log('accuracy', accuracy) # 精度を'accuracy'の名前でログをとる。

    # 学習に用いるオプティマイザを返す関数を書く。
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

# 6-15: 学習時にモデルの重みを保存する条件を指定
checkpoint = pl.callbacks.ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_top_k=1,
    save_weights_only=True,
    dirpath='model/',
)

# 学習の方法を指定
trainer = pl.Trainer(
    gpus=1, 
    max_epochs=10,
    callbacks = [checkpoint]
)
# 6-16: PyTorch Lightningを使用してBERTモデルを初期化し、分類のための設定を行う
model = BertForSequenceClassification_pl(
    MODEL_NAME, num_labels=9, lr=1e-5
)

# ファインチューニングを行う。
trainer.fit(model, dataloader_train, dataloader_val)  # モデルを訓練データと検証データを使ってファインチューニングする

# 6-17
best_model_path = checkpoint.best_model_path # ベストモデルのファイル
print('ベストモデルのファイル: ', checkpoint.best_model_path)  # 訓練中に最も良い性能を示したモデルのパスを表示する
print('ベストモデルの検証データに対する損失: ', checkpoint.best_model_score)  # ベストモデルの検証データに対する損失を表示する

# 6-18
# TensorBoard という視覚化ツールの Jupyter Notebook 拡張を読み込む
#// %load_ext tensorboard
# TensorBoard を起動して、特定のディレクトリに保存されたログファイルを表示する
#// %tensorboard --logdir ./

# 6-19
test = trainer.test(dataloaders=dataloader_test)  # 訓練済みモデルをテストデータセットで評価する
print(f'Accuracy: {test[0]["accuracy"]:.2f}')  # テストデータセットでのモデルの精度を表示する

# 6-20: 保存されたベストモデルのチェックポイントをロードする
model = BertForSequenceClassification_pl.load_from_checkpoint(
    best_model_path
)

# Transformers対応のモデルを./model_transformesに保存
model.bert_sc.save_pretrained('./model_transformers')  # PyTorch LightningモデルをTransformersライブラリ形式で保存する

# 6-21: 保存されたTransformers形式のモデルをロードする
bert_sc = BertForSequenceClassification.from_pretrained(
    './model_transformers'
)
