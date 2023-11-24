# 7.文章の抽出

# カレントディレクトリを "./chap7/" にする
import os
file_directory = os.path.dirname(os.path.abspath(__file__)) # fileのあるディレクトリのパスを取得
target_directory = os.path.join(file_directory, 'chap7') # './chap7/'へのパスを構築
os.chdir(target_directory) # カレントディレクトリを変更

# 7-1
#// !mkdir chap7
#// %cd ./chap7

# 7-2
#// !pip install transformers==4.18.0 fugashi==1.1.0 ipadic==1.0.0 pytorch-lightning==1.6.1

# 7-3
import random
import glob
import json
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer, BertModel
import pytorch_lightning as pl

# 日本語の事前学習モデル
MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'

# 7-4
class BertForSequenceClassificationMultiLabel(torch.nn.Module):
    
    def __init__(self, model_name, num_labels):
        super().__init__()
        # BertModelのロード
        self.bert = BertModel.from_pretrained(model_name) 
        # 線形変換を初期化しておく
        self.linear = torch.nn.Linear(
            self.bert.config.hidden_size, num_labels
        ) 

    def forward(
        self, 
        input_ids=None, 
        attention_mask=None, 
        token_type_ids=None, 
        labels=None
    ):
        # データを入力しBERTの最終層の出力を得る。
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids)
        last_hidden_state = bert_output.last_hidden_state
        
        # [PAD]以外のトークンで隠れ状態の平均をとる
        averaged_hidden_state = \
            (last_hidden_state*attention_mask.unsqueeze(-1)).sum(1) \
            / attention_mask.sum(1, keepdim=True)
        
        # 線形変換
        scores = self.linear(averaged_hidden_state) 
        
        # 出力の形式を整える。
        output = {'logits': scores}

        # labelsが入力に含まれていたら、損失を計算し出力する。
        if labels is not None: 
            loss = torch.nn.BCEWithLogitsLoss()(scores, labels.float())
            output['loss'] = loss
            
        # 属性でアクセスできるようにする。
        output = type('bert_output', (object,), output) 

        return output

# 7-5
tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)
bert_scml = BertForSequenceClassificationMultiLabel(
    MODEL_NAME, num_labels=2
) 
bert_scml = bert_scml.cuda()

# 7-6
text_list = [
    '今日の仕事はうまくいったが、体調があまり良くない。',
    '昨日は楽しかった。'
]

labels_list = [
    [1, 1],
    [0, 1]
]

# データの符号化
encoding = tokenizer(
    text_list, 
    padding='longest',  
    return_tensors='pt'
)
encoding = { k: v.cuda() for k, v in encoding.items() }
labels = torch.tensor(labels_list).cuda()

# BERTへデータを入力し分類スコアを得る。
with torch.no_grad():
    output = bert_scml(**encoding)
scores = output.logits

# スコアが正ならば、そのカテゴリーを選択する。
labels_predicted = ( scores > 0 ).int()

# 精度の計算
num_correct = ( labels_predicted == labels ).all(-1).sum().item()
accuracy = num_correct/labels.size(0)

# 7-7
# データの符号化
encoding = tokenizer(
    text_list, 
    padding='longest',  
    return_tensors='pt'
)
encoding['labels'] = torch.tensor(labels_list) # 入力にlabelsを含める。
encoding = { k: v.cuda() for k, v in encoding.items() }

output = bert_scml(**encoding)
loss = output.loss # 損失

# 7-8
# データのダウンロード
#// !wget https://s3-ap-northeast-1.amazonaws.com/dev.tech-sketch.jp/chakki/public/chABSA-dataset.zip
# データの解凍
#// !unzip chABSA-dataset.zip 

# 7-9
with open('chABSA-dataset/e00030_ann.json', encoding='utf-8') as f:
    data = json.load(f)
#// data = json.load(open('chABSA-dataset/e00030_ann.json'))
print( data['sentences'][0] )

# 7-10
category_id = {'negative':0, 'neutral':1 , 'positive':2}

dataset = []
for file in glob.glob('chABSA-dataset/*.json'):
    with open(file, encoding='utf-8') as f:
        data = json.load(f)
#//     data = json.load(open(file))
    # 各データから文章（text）を抜き出し、ラベル（'labels'）を作成
    for sentence in data['sentences']:
        text = sentence['sentence'] 
        labels = [0,0,0]
        for opinion in sentence['opinions']:
            labels[category_id[opinion['polarity']]] = 1
        sample = {'text': text, 'labels': labels}
        dataset.append(sample)

# 7-11
print(dataset[0])

# 7-12
# トークナイザのロード
tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)

# 各データの形式を整える
max_length = 128
dataset_for_loader = []
for sample in dataset:
    text = sample['text']
    labels = sample['labels']
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True
    )
    encoding['labels'] = labels
    encoding = { k: torch.tensor(v) for k, v in encoding.items() }
    dataset_for_loader.append(encoding)

# データセットの分割
random.shuffle(dataset_for_loader) 
n = len(dataset_for_loader)
n_train = int(0.6*n)
n_val = int(0.2*n)
dataset_train = dataset_for_loader[:n_train] # 学習データ
dataset_val = dataset_for_loader[n_train:n_train+n_val] # 検証データ
dataset_test = dataset_for_loader[n_train+n_val:] # テストデータ

#　データセットからデータローダを作成
dataloader_train = DataLoader(
    dataset_train, batch_size=32, shuffle=True
) 
dataloader_val = DataLoader(dataset_val, batch_size=256)
dataloader_test = DataLoader(dataset_test, batch_size=256)

# 7-13
class BertForSequenceClassificationMultiLabel_pl(pl.LightningModule):

    def __init__(self, model_name, num_labels, lr):
        super().__init__()
        self.save_hyperparameters() 
        self.bert_scml = BertForSequenceClassificationMultiLabel(
            model_name, num_labels=num_labels
        ) 

    def training_step(self, batch, batch_idx):
        output = self.bert_scml(**batch)
        loss = output.loss
        self.log('train_loss', loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        output = self.bert_scml(**batch)
        val_loss = output.loss
        self.log('val_loss', val_loss)

    def test_step(self, batch, batch_idx):
        labels = batch.pop('labels')
        output = self.bert_scml(**batch)
        scores = output.logits
        labels_predicted = ( scores > 0 ).int()
        num_correct = ( labels_predicted == labels ).all(-1).sum().item()
        accuracy = num_correct/scores.size(0)
        self.log('accuracy', accuracy)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

checkpoint = pl.callbacks.ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_top_k=1,
    save_weights_only=True,
    dirpath='model/',
)

trainer = pl.Trainer(
    gpus=1, 
    max_epochs=5,
    callbacks = [checkpoint]
)

model = BertForSequenceClassificationMultiLabel_pl(
    MODEL_NAME, 
    num_labels=3, 
    lr=1e-5
)
trainer.fit(model, dataloader_train, dataloader_val)
test = trainer.test(dataloaders=dataloader_test)
print(f'Accuracy: {test[0]["accuracy"]:.2f}')

# 7-14
# 入力する文章
text_list = [
    "今期は売り上げが順調に推移したが、株価は低迷の一途を辿っている。",
    "昨年から黒字が減少した。",
    "今日の飲み会は楽しかった。"
]

# モデルのロード
best_model_path = checkpoint.best_model_path
model = BertForSequenceClassificationMultiLabel_pl.load_from_checkpoint(best_model_path)
bert_scml = model.bert_scml.cuda()

# データの符号化
encoding = tokenizer(
    text_list, 
    padding = 'longest',
    return_tensors='pt'
)
encoding = { k: v.cuda() for k, v in encoding.items() }

# BERTへデータを入力し分類スコアを得る。
with torch.no_grad():
    output = bert_scml(**encoding)
scores = output.logits
labels_predicted = ( scores > 0 ).int().cpu().numpy().tolist()

# 結果を表示
for text, label in zip(text_list, labels_predicted):
    print('--')
    print(f'入力：{text}')
    print(f'出力：{label}')
