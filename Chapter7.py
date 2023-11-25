# 7.文章の抽出
# 実行方法："chap7"フォルダを作成し、"https://s3-ap-northeast-1.amazonaws.com/dev.tech-sketch.jp/chakki/public/chABSA-dataset.zip"を解凍して、VSCodeでChapter7.pyを実行

# 7-1: カレントディレクトリを "./chap7/" に設定
import os
file_directory = os.path.dirname(os.path.abspath(__file__)) # fileのあるディレクトリのパスを取得。カレントディレクトリを取得
target_directory = os.path.join(file_directory, 'chap7') # './chap7/'へのパスを構築。目標ディレクトリへのパスを結合
os.chdir(target_directory) # カレントディレクトリを変更。カレントディレクトリを変更する
#// !mkdir chap7
#// %cd ./chap7

# 7-2: 必要なライブラリのインストール
#// !pip install transformers==4.18.0 fugashi==1.1.0 ipadic==1.0.0 pytorch-lightning==1.6.1

# 7-3: 必要なライブラリをインポートし、BERTの日本語モデルを設定
import random
import glob
import json
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer, BertModel
import pytorch_lightning as pl

MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking' # 日本語の事前学習モデルを設定。日本語のBERTモデル名を設定

# 7-4: BERTを用いた多ラベル分類モデルの定義
class BertForSequenceClassificationMultiLabel(torch.nn.Module):
    
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name) # BertModelのロード。事前学習済みのBERTモデルをロード
        self.linear = torch.nn.Linear(self.bert.config.hidden_size, num_labels) # 線形変換を初期化しておく。出力層を初期化

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        # データを入力しBERTの最終層の出力を得る。
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids) # データを入力しBERTの最終層の出力を得る
        last_hidden_state = bert_output.last_hidden_state
        # [PAD]以外のトークンで隠れ状態の平均をとる
        averaged_hidden_state = (last_hidden_state*attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True) # [PAD]以外のトークンで隠れ状態の平均をとる
        scores = self.linear(averaged_hidden_state) # 線形変換。出力層を通してスコアを計算
        output = {'logits': scores} # 出力の形式を整える。出力形式を辞書形式に整理

        if labels is not None:
            loss = torch.nn.BCEWithLogitsLoss()(scores, labels.float()) # labelsが入力に含まれていたら、損失を計算し出力する。損失を計算
            output['loss'] = loss
            
        output = type('bert_output', (object,), output) # 属性でアクセスできるようにする。出力をオブジェクト形式に変換

        return output

# 7-5: トークナイザとモデルの初期化
tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME) # トークナイザの初期化。日本語用のトークナイザを初期化
bert_scml = BertForSequenceClassificationMultiLabel(MODEL_NAME, num_labels=2) # モデルの初期化。多ラベル分類モデルを初期化
bert_scml = bert_scml.cuda() # モデルをGPUに移動。モデルをGPU対応に設定

# 7-6: データの準備とモデルによる予測
text_list = [
    '今日の仕事はうまくいったが、体調があまり良くない。',
    '昨日は楽しかった。'
]

labels_list = [
    [1, 1],
    [0, 1]
]

# データの符号化
encoding = tokenizer(text_list, padding='longest', return_tensors='pt') # データの符号化。テキストを符号化
encoding = { k: v.cuda() for k, v in encoding.items() } # データをGPUに移動。符号化されたデータをGPUに移動
labels = torch.tensor(labels_list).cuda() # ラベルをGPUに移動。ラベルをテンソルに変換しGPUに移動

# BERTへデータを入力し分類スコアを得る
with torch.no_grad():
    output = bert_scml(**encoding) # BERTモデルにデータを入力
scores = output.logits # 分類スコアを取得

# スコアが正ならば、そのカテゴリーを選択する
labels_predicted = (scores > 0).int() # 予測ラベルの決定。スコアが正の場合、該当カテゴリーを選択

# 精度の計算
num_correct = (labels_predicted == labels).all(-1).sum().item() # 正解数の計算。予測ラベルと実際のラベルが一致する数を計算
accuracy = num_correct/labels.size(0) # 精度の計算。全データ数で割って精度を計算

# 7-7: 損失の計算
# データの符号化
encoding = tokenizer(text_list, padding='longest', return_tensors='pt') # データの符号化。テキストを符号化
encoding['labels'] = torch.tensor(labels_list) # 入力にlabelsを含める。ラベルを符号化データに追加
encoding = { k: v.cuda() for k, v in encoding.items() } # データをGPUに移動。符号化されたデータをGPUに移動

output = bert_scml(**encoding) # BERTモデルにデータを入力
loss = output.loss # 損失の計算。計算された損失を取得

# 7-8: データのダウンロードと解凍
# データのダウンロード
#// !wget https://s3-ap-northeast-1.amazonaws.com/dev.tech-sketch.jp/chakki/public/chABSA-dataset.zip
# データの解凍
#// !unzip chABSA-dataset.zip 

# 7-9: JSONファイルの読み込みと表示
with open('chABSA-dataset/e00030_ann.json', encoding='utf-8') as f:
    data = json.load(f) # JSONファイルを読み込む。指定されたJSONファイルを開いてデータを読み込む
print( data['sentences'][0] ) # データの一部を表示。読み込んだデータの最初の文を表示

# 7-10: データセットの作成
category_id = {'negative':0, 'neutral':1 , 'positive':2} # カテゴリーIDの設定。感情カテゴリーに対応するIDを設定

dataset = [] # データセットを初期化
for file in glob.glob('chABSA-dataset/*.json'): # JSONファイルを走査
    with open(file, encoding='utf-8') as f:
        data = json.load(f) # JSONファイルを読み込む。各JSONファイルを開いてデータを読み込む
    # 各データから文章（text）を抜き出し、ラベル（'labels'）を作成
    for sentence in data['sentences']: # 各文に対してループ
        text = sentence['sentence'] # 文を取り出す。文の内容を取得
        labels = [0,0,0] # ラベルを初期化
        for opinion in sentence['opinions']: # 各意見に対してループ
            labels[category_id[opinion['polarity']]] = 1 # ラベルを設定。意見の感情極性に基づいてラベルを更新
        sample = {'text': text, 'labels': labels} # サンプルを作成。文とラベルからサンプルを作成
        dataset.append(sample) # サンプルをデータセットに追加

# 7-11: データセットの一部を表示
print(dataset[0]) # データセットの最初のサンプルを表示。データセットの最初のサンプルを表示

# 7-12: データセットの前処理とデータローダの作成
tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME) # トークナイザのロード。トークナイザを初期化

# 各データの形式を整える
max_length = 128 # 最大長の設定。トークナイズの最大長を設定
dataset_for_loader = [] # データローダ用のデータセットを初期化
for sample in dataset: # データセットの各サンプルに対してループ
    text = sample['text'] # 文を取得
    labels = sample['labels'] # ラベルを取得
    encoding = tokenizer(text, max_length=max_length, padding='max_length', truncation=True) # 文を符号化。文をトークナイズし、指定された長さに合わせてパディングや切り捨てを行う
    encoding['labels'] = labels # 符号化されたデータにラベルを追加
    encoding = { k: torch.tensor(v) for k, v in encoding.items() } # テンソルに変換。符号化されたデータをテンソルに変換
    dataset_for_loader.append(encoding) # データローダ用のデータセットに追加

# データセットの分割
random.shuffle(dataset_for_loader) # データセットをシャッフル。データセットをランダムにシャッフル
n = len(dataset_for_loader) # データセットの総数を取得
n_train = int(0.6*n) # 訓練データの数を計算。データセットの60%を訓練データとして設定
n_val = int(0.2*n) # 検証データの数を計算。データセットの20%を検証データとして設定
dataset_train = dataset_for_loader[:n_train] # 学習データを取得。データセットの最初から訓練データの数だけを訓練データとして取得
dataset_val = dataset_for_loader[n_train:n_train+n_val] # 検証データを取得。訓練データの次から検証データの数だけを検証データとして取得
dataset_test = dataset_for_loader[n_train+n_val:] # テストデータを取得。残りのデータをテストデータとして取得

# データセットからデータローダを作成
dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True) # 訓練データローダを作成。訓練データセットからバッチサイズ32でシャッフルされたデータローダを作成
dataloader_val = DataLoader(dataset_val, batch_size=256) # 検証データローダを作成。検証データセットからバッチサイズ256でデータローダを作成
dataloader_test = DataLoader(dataset_test, batch_size=256) # テストデータローダを作成。テストデータセットからバッチサイズ256でデータローダを作成

# 7-13: BERTに基づくマルチラベル分類モデルの定義とトレーニング
class BertForSequenceClassificationMultiLabel_pl(pl.LightningModule):
    # コンストラクタ：BERTモデルの設定
    def __init__(self, model_name, num_labels, lr):
        super().__init__()
        self.save_hyperparameters()  # ハイパーパラメータの保存
        self.bert_scml = BertForSequenceClassificationMultiLabel(
            model_name, num_labels=num_labels
        )  # BERTモデルの初期化

    # トレーニングステップ：モデルのトレーニング処理
    def training_step(self, batch, batch_idx):
        output = self.bert_scml(**batch)  # バッチデータの処理
        loss = output.loss  # 損失の計算
        self.log('train_loss', loss)  # トレーニング損失の記録
        return loss  # 損失の返却
        
    # バリデーションステップ：モデルの評価処理
    def validation_step(self, batch, batch_idx):
        output = self.bert_scml(**batch)  # バッチデータの処理
        val_loss = output.loss  # 損失の計算
        self.log('val_loss', val_loss)  # バリデーション損失の記録

    # テストステップ：モデルのテスト処理
    def test_step(self, batch, batch_idx):
        labels = batch.pop('labels')  # ラベルの取得
        output = self.bert_scml(**batch)  # バッチデータの処理
        scores = output.logits  # スコアの計算
        labels_predicted = (scores > 0).int()  # ラベルの予測
        num_correct = (labels_predicted == labels).all(-1).sum().item()  # 正答数の計算
        accuracy = num_correct/scores.size(0)  # 精度の計算
        self.log('accuracy', accuracy)  # 精度の記録

    # オプティマイザの設定
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)  # Adamオプティマイザの設定

# モデルチェックポイントの設定
checkpoint = pl.callbacks.ModelCheckpoint(
    monitor='val_loss',  # モニタリングする値
    mode='min',  # 最小化モード
    save_top_k=1,  # 保存するモデルの数
    save_weights_only=True,  # 重みのみ保存
    dirpath='model/',  # 保存先のディレクトリパス
)

# トレーナーの設定
trainer = pl.Trainer(
    gpus=1,  # GPUの数
    max_epochs=5,  # 最大エポック数
    callbacks=[checkpoint]  # コールバックの設定
)

# モデルの初期化
model = BertForSequenceClassificationMultiLabel_pl(
    MODEL_NAME, 
    num_labels=3, 
    lr=1e-5
)
# モデルのトレーニング
trainer.fit(model, dataloader_train, dataloader_val)
# モデルのテスト
test = trainer.test(dataloaders=dataloader_test)
# 精度の表示
print(f'Accuracy: {test[0]["accuracy"]:.2f}')

# 7-14: BERTモデルを使ってテキストデータの分類
# 入力する文章
text_list = [
    "今期は売り上げが順調に推移したが、株価は低迷の一途を辿っている。",
    "昨年から黒字が減少した。",
    "今日の飲み会は楽しかった。"
]

# モデルのロード
best_model_path = checkpoint.best_model_path  # 最良モデルのパス取得
model = BertForSequenceClassificationMultiLabel_pl.load_from_checkpoint(best_model_path)  # モデルのロード
bert_scml = model.bert_scml.cuda()  # モデルをGPUに移動

# データの符号化
encoding = tokenizer(
    text_list, 
    padding='longest',  # 最長バッチのパディング
    return_tensors='pt'  # PyTorchテンソル形式での返却
)
encoding = {k: v.cuda() for k, v in encoding.items()}  # データをGPUに移動

# BERTへデータを入力し分類スコアを得る
with torch.no_grad():
    output = bert_scml(**encoding)  # BERTモデルにデータを入力
scores = output.logits  # スコアの取得
labels_predicted = (scores > 0).int().cpu().numpy().tolist()  # ラベルの予測

# 結果を表示
for text, label in zip(text_list, labels_predicted):
    print('--')
    print(f'入力：{text}')  # 入力テキストの表示
    print(f'出力：{label}')  # 予測ラベルの表示
