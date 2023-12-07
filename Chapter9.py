# 9.文章校正
# 実行方法："chap9"フォルダを作成し、"https://nlp.ist.i.kyoto-u.ac.jp/DLcounter/lime.cgi?down=https://nlp.ist.i.kyoto-u.ac.jp/nl-resource/JWTD/jwtd.tar.gz&name=JWTD.tar.gz"を解凍して、VSCodeでChapter9.pyを実行

# カレントディレクトリを "./chap9/" にする
import os
file_directory = os.path.dirname(os.path.abspath(__file__)) # fileのあるディレクトリのパスを取得
target_directory = os.path.join(file_directory, 'chap9') # './chap9/'へのパスを構築
os.chdir(target_directory) # カレントディレクトリを変更

# 9-1: 'chap9'という名前の新しいフォルダを作成し、カレントディレクトリをそのフォルダに変更
#// !mkdir chap9
#// %cd ./chap9

# 9-2: 必要なライブラリ(transformers, fugashi, ipadic, pytorch-lightning)を特定のバージョンでインストール
#// !pip install transformers==4.18.0 fugashi==1.1.0 ipadic==1.0.0 pytorch-lightning==1.6.1

# 9-3: 必要なライブラリをインポート
import random
from tqdm import tqdm
import unicodedata

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer, BertForMaskedLM
import pytorch_lightning as pl

# 日本語の事前学習済みモデル
MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'

# 9-4: SC_tokenizerクラスの定義。BertJapaneseTokenizerを継承。
class SC_tokenizer(BertJapaneseTokenizer):
    def encode_plus_tagged(self, wrong_text, correct_text, max_length=128):
        """
        ファインチューニング時に使用。
        誤変換を含む文章と正しい文章を入力とし、
        符号化を行いBERTに入力できる形式にする。
        """
        # 誤変換した文章をトークン化し、符号化する。
        encoding = self(
            wrong_text, 
            max_length=max_length, 
            padding='max_length', 
            truncation=True
        )
        # 正しい文章をトークン化し、符号化する。
        encoding_correct = self(
            correct_text,
            max_length=max_length,
            padding='max_length',
            truncation=True
        ) 
        # 正しい文章の符号化されたIDをラベルとして設定する。
        encoding['labels'] = encoding_correct['input_ids'] 

        return encoding

    def encode_plus_untagged(self, text, max_length=None, return_tensors=None):
        """
        文章を符号化し、それぞれのトークンの文章中の位置も特定しておく。
        """
        # 文章をトークン化し、それぞれのトークンと元の文字列を対応づける。
        tokens = []  # トークンを追加していくリスト。
        tokens_original = []  # トークンに対応する元の文字列を追加していくリスト。
        words = self.word_tokenizer.tokenize(text)  # 文章をMeCabで単語に分割。
        for word in words:
            # 単語をサブワードに分割する。
            tokens_word = self.subword_tokenizer.tokenize(word) 
            tokens.extend(tokens_word)
            if tokens_word[0] == '[UNK]':  # 未知語に対応。
                tokens_original.append(word)
            else:
                tokens_original.extend([
                    token.replace('##', '') for token in tokens_word
                ])

        # 各トークンの文章中での位置を調べる。（空白の位置を考慮する）
        position = 0
        spans = []  # トークンの位置情報を追加していくリスト。
        for token in tokens_original:
            l = len(token)
            while True:
                if token != text[position:position + l]:
                    position += 1
                else:
                    spans.append([position, position + l])
                    position += l
                    break

        # トークンを符号化し、BERTに入力できる形式にする。
        input_ids = self.convert_tokens_to_ids(tokens) 
        encoding = self.prepare_for_model(
            input_ids, 
            max_length=max_length, 
            padding='max_length' if max_length else False, 
            truncation=True if max_length else False
        )
        sequence_length = len(encoding['input_ids'])
        # 特殊トークン[CLS]に対するダミーのspanを追加。
        spans = [[-1, -1]] + spans[:sequence_length - 2]
        # 特殊トークン[SEP]、[PAD]に対するダミーのspanを追加。
        spans = spans + [[-1, -1]] * (sequence_length - len(spans))

        # 必要に応じてtorch.Tensorに変換する。
        if return_tensors == 'pt':
            encoding = {k: torch.tensor([v]) for k, v in encoding.items()}

        return encoding, spans

    def convert_bert_output_to_text(self, text, labels, spans):
        """
        推論時に使用。
        文章と、各トークンのラベルの予測値、文章中での位置を入力とする。
        そこから、BERTによって予測された文章に変換。
        """
        assert len(spans) == len(labels)  # spansとlabelsの長さが一致することを確認。

        # 特殊トークンに対応する部分をlabelsとspansから除外。
        labels = [label for label, span in zip(labels, spans) if span[0] != -1]
        spans = [span for span in spans if span[0] != -1]

        # BERTが予測した文章を作成する。
        predicted_text = ''
        position = 0
        for label, span in zip(labels, spans):
            start, end = span
            if position != start:  # 空白を処理。
                predicted_text += text[position:start]
            predicted_token = self.convert_ids_to_tokens(label)
            predicted_token = predicted_token.replace('##', '')
            predicted_token = unicodedata.normalize('NFKC', predicted_token)
            predicted_text += predicted_token
            position = end
        
        return predicted_text

# 9-5: モデル名を用いてトークナイザーを初期化する
tokenizer = SC_tokenizer.from_pretrained(MODEL_NAME)  # MODEL_NAMEを使用して、SC_tokenizerを事前に訓練された状態で初期化

# 9-6: 誤ったテキストと正しいテキストを用いてエンコーディングを行う
wrong_text = '優勝トロフィーを変換した'  # 誤ったテキストの例
correct_text = '優勝トロフィーを返還した'  # 正しいテキストの例
encoding = tokenizer.encode_plus_tagged(
    wrong_text, correct_text, max_length=12
)  # 誤ったテキストと正しいテキストをエンコードする
print(encoding)  # エンコーディングの結果を出力

# 9-7: 誤ったテキストを用いてエンコーディングし、スパン情報を取得する
wrong_text = '優勝トロフィーを変換した'  # 誤ったテキストの例
encoding, spans = tokenizer.encode_plus_untagged(
    wrong_text, return_tensors='pt'
)  # 誤ったテキストをエンコードし、テンソルを返す
print('# encoding')  # エンコーディングの出力をラベル付け
print(encoding)  # エンコーディングの結果を出力
print('# spans')  # スパン情報の出力をラベル付け
print(spans)  # スパン情報を出力

# 9-8: 予測されたラベルを用いてテキストを変換する
predicted_labels = [2, 759, 18204, 11, 8274, 15, 10, 3]  # 予測されたラベルの例
predicted_text = tokenizer.convert_bert_output_to_text(
    wrong_text, predicted_labels, spans
)  # 誤ったテキスト、予測ラベル、スパンを用いて修正されたテキストを生成
print(predicted_text)  # 生成されたテキストを出力

# 9-9: BERTモデルを初期化し、GPUに転送する
bert_mlm = BertForMaskedLM.from_pretrained(MODEL_NAME)  # MODEL_NAMEを使用して、BertForMaskedLMを事前に訓練された状態で初期化
bert_mlm = bert_mlm.cuda()  # モデルをGPUに転送

# 9-10: テキストの誤りをBERTモデルを用いて訂正する

# 訂正するテキストの定義
text = '優勝トロフィーを変換した。'

# テキストを符号化し、各トークンの文章中の位置を計算
encoding, spans = tokenizer.encode_plus_untagged(
    text, return_tensors='pt'
)
encoding = { k: v.cuda() for k, v in encoding.items() }  # エンコーディングされたデータをGPUに移動

# BERTモデルに入力し、トークン毎にスコアが最も高いトークンのIDを予測
with torch.no_grad():  # 勾配計算を無効化
    output = bert_mlm(**encoding)  # BERTモデルにエンコーディングされたデータを入力
    scores = output.logits  # ロジット（各トークンのスコア）を取得
    labels_predicted = scores[0].argmax(-1).cpu().numpy().tolist()  # 各トークンに対する最も高いスコアのラベルIDを予測

# 予測されたラベル列を元の文章に変換
predict_text = tokenizer.convert_bert_output_to_text(
    text, labels_predicted, spans
)  # 予測されたラベルとスパン情報を用いて、修正されたテキストを生成

# 9-11: テキストの誤りを訂正するためのBERTモデルのトレーニングデータとプロセス

# 訓練データの定義（誤ったテキストと正しいテキストのペア）
data = [
    {
        'wrong_text': '優勝トロフィーを変換した。',
        'correct_text': '優勝トロフィーを返還した。',
    },
    {
        'wrong_text': '人と森は強制している。',
        'correct_text': '人と森は共生している。',
    }
]

# 各データを符号化し、データローダへ入力できるようにする。
max_length=32  # 最大トークン長の設定
dataset_for_loader = []  # データローダ用のデータセットを初期化
for sample in data:
    wrong_text = sample['wrong_text']  # 誤ったテキスト
    correct_text = sample['correct_text']  # 正しいテキスト
    encoding = tokenizer.encode_plus_tagged(
        wrong_text, correct_text, max_length=max_length
    )  # テキストペアを符号化
    encoding = { k: torch.tensor(v) for k, v in encoding.items() }  # 各符号化要素をテンソルに変換
    dataset_for_loader.append(encoding)  # データセットに追加

# データローダを作成
dataloader = DataLoader(dataset_for_loader, batch_size=2)  # データローダをバッチサイズ2で初期化

# ミニバッチをBERTモデルへ入力し、損失を計算。
for batch in dataloader:
    encoding = { k: v.cuda() for k, v in batch.items() }  # バッチデータをGPUに移動
    output = bert_mlm(**encoding)  # BERTモデルにエンコーディングされたデータを入力
    loss = output.loss  # モデルの出力から損失を計算

# 9-12
#// !curl -L "https://nlp.ist.i.kyoto-u.ac.jp/DLcounter/lime.cgi?down=https://nlp.ist.i.kyoto-u.ac.jp/nl-resource/JWTD/jwtd.tar.gz&name=JWTD.tar.gz" -o JWTD.tar.gz
#// !tar zxvf JWTD.tar.gz

# 9-13: 漢字の誤変換データセットの作成と準備
def create_dataset(data_df):
    # トークナイザーの初期化
    tokenizer = SC_tokenizer.from_pretrained(MODEL_NAME)  # MODEL_NAMEを用いてSC_tokenizerを初期化

    def check_token_count(row):
        """
        誤変換の文章と正しい文章でトークンに対応がつくかどうかを判定。
        （条件は上の文章を参照）
        """
        # 誤変換のテキストと正しいテキストをトークン化
        wrong_text_tokens = tokenizer.tokenize(row['wrong_text'])
        correct_text_tokens = tokenizer.tokenize(row['correct_text'])

        # トークン数が異なる場合はFalseを返す
        if len(wrong_text_tokens) != len(correct_text_tokens):
            return False
        
        # トークンの違いの数をカウント
        diff_count = 0
        threthold_count = 2  # 許容される最大のトークンの違いの数
        for wrong_text_token, correct_text_token in zip(wrong_text_tokens, correct_text_tokens):
            if wrong_text_token != correct_text_token:
                diff_count += 1
                if diff_count > threthold_count:  # 違いが閾値を超えた場合はFalseを返す
                    return False
        return True  # トークンの違いが閾値以内であればTrueを返す

    def normalize(text):
        """
        文字列の正規化
        """
        text = text.strip()  # 空白を削除
        text = unicodedata.normalize('NFKC', text)  # NFKC正規化を行う
        return text

    # 漢字の誤変換のデータのみを抜き出す。
    category_type = 'kanji-conversion'  # カテゴリタイプを指定
    data_df.query('category == @category_type', inplace=True)  # 漢字の誤変換に関するデータのみ抽出
    data_df.rename(columns={'pre_text': 'wrong_text', 'post_text': 'correct_text'}, inplace=True)  # 列名を変更
    
    # 誤変換と正しい文章をそれぞれ正規化し、
    # それらの間でトークン列に対応がつくもののみを抜き出す。
    data_df['wrong_text'] = data_df['wrong_text'].map(normalize)  # 誤ったテキストの正規化
    data_df['correct_text'] = data_df['correct_text'].map(normalize)  # 正しいテキストの正規化
    kanji_conversion_num = len(data_df)  # 元のデータ数
    data_df = data_df[data_df.apply(check_token_count, axis=1)]  # トークンの対応がつくデータのみ抽出
    same_tokens_count_num = len(data_df)  # 抽出後のデータ数
    # 結果の表示
    print(
        f'- 漢字誤変換の総数：{kanji_conversion_num}',
        f'- トークンの対応関係のつく文章の総数: {same_tokens_count_num}',
        f'  (全体の{same_tokens_count_num/kanji_conversion_num*100:.0f}%)',
        sep='\n'
    )
    return data_df[['wrong_text', 'correct_text']].to_dict(orient='records')  # データセットを辞書形式で返す

# データのロード
# 学習用データのロード
train_df = pd.read_json(
    './jwtd/train.jsonl', orient='records', lines=True
)
# テスト用データのロード
test_df = pd.read_json(
    './jwtd/test.jsonl', orient='records', lines=True
)

# 学習用と検証用データ
print('学習と検証用のデータセット：')
dataset = create_dataset(train_df)  # 学習用データセットの作成
random.shuffle(dataset)  # データセットのシャッフル
n = len(dataset)  # データセットの総数
n_train = int(n*0.8)  # 学習用データの数
dataset_train = dataset[:n_train]  # 学習用データセット
dataset_val = dataset[n_train:]  # 検証用データセット

# テストデータ
print('テスト用のデータセット：')
dataset_test = create_dataset(test_df)  # テスト用データセットの作成

# 9-14: データセットをデータローダに入力可能な形式に変換し、データローダを作成

def create_dataset_for_loader(tokenizer, dataset, max_length):
    """
    データセットをデータローダに入力可能な形式にする。
    """
    dataset_for_loader = []  # データローダ用データセットを格納するリスト
    for sample in tqdm(dataset):
        wrong_text = sample['wrong_text']  # 誤ったテキスト
        correct_text = sample['correct_text']  # 正しいテキスト
        encoding = tokenizer.encode_plus_tagged(
            wrong_text, correct_text, max_length=max_length
        )  # テキストを符号化
        encoding = { k: torch.tensor(v) for k, v in encoding.items() }  # 符号化されたデータをテンソルに変換
        dataset_for_loader.append(encoding)  # データセットに追加
    return dataset_for_loader  # 変換されたデータセットを返す

# トークナイザーの初期化
tokenizer = SC_tokenizer.from_pretrained(MODEL_NAME)  # MODEL_NAMEを用いてSC_tokenizerを初期化

# データセットの作成
max_length = 32  # 最大トークン長を指定
dataset_train_for_loader = create_dataset_for_loader(
    tokenizer, dataset_train, max_length
)  # 訓練用データセットの作成
dataset_val_for_loader = create_dataset_for_loader(
    tokenizer, dataset_val, max_length
)  # 検証用データセットの作成

# データローダの作成
dataloader_train = DataLoader(
    dataset_train_for_loader, batch_size=32, shuffle=True
)  # 訓練用データローダをバッチサイズ32で作成し、シャッフルを有効にする
dataloader_val = DataLoader(dataset_val_for_loader, batch_size=256)  # 検証用データローダをバッチサイズ256で作成

# 9-15: PyTorch Lightningを使用したBERTモデルのファインチューニング
class BertForMaskedLM_pl(pl.LightningModule):
    # BERTモデルのPyTorch Lightningクラス
    def __init__(self, model_name, lr):
        super().__init__()
        self.save_hyperparameters()  # ハイパーパラメータを保存
        self.bert_mlm = BertForMaskedLM.from_pretrained(model_name)  # BERTモデルの初期化
        
    def training_step(self, batch, batch_idx):
        output = self.bert_mlm(**batch)  # バッチをモデルに入力し、出力を取得
        loss = output.loss  # 損失を計算
        self.log('train_loss', loss)  # 訓練の損失を記録
        return loss  # 損失を返す
        
    def validation_step(self, batch, batch_idx):
        output = self.bert_mlm(**batch)  # 検証のバッチをモデルに入力し、出力を取得
        val_loss = output.loss  # 検証の損失を計算
        self.log('val_loss', val_loss)  # 検証の損失を記録
   
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)  # 最適化関数の設定

# モデルのチェックポイントの設定
checkpoint = pl.callbacks.ModelCheckpoint(
    monitor='val_loss',  # 監視する値は検証の損失
    mode='min',  # 損失が最小のモデルを保存
    save_top_k=1,  # トップのモデル1個を保存
    save_weights_only=True,  # 重みのみを保存
    dirpath='model/'  # 保存先のディレクトリ
)

# トレーナーの設定
trainer = pl.Trainer(
    gpus=1,  # GPUの使用設定（1つ使用）
    max_epochs=5,  # 最大エポック数
    callbacks=[checkpoint]  # コールバックの設定（チェックポイント）
)

# ファインチューニング
model = BertForMaskedLM_pl(MODEL_NAME, lr=1e-5)  # モデルの初期化
trainer.fit(model, dataloader_train, dataloader_val)  # トレーニングの実行
best_model_path = checkpoint.best_model_path  # 最良モデルのパスを取得

# 9-16: BERTモデルを用いたテキストの誤りの修正
def predict(text, tokenizer, bert_mlm):
    """
    文章を入力として受け、BERTが予測した文章を出力
    """
    # 符号化: テキストをBERTモデルが処理できる形式に変換
    encoding, spans = tokenizer.encode_plus_untagged(
        text, return_tensors='pt'
    ) 
    encoding = { k: v.cuda() for k, v in encoding.items() }  # 符号化されたデータをGPUに移動

    # ラベルの予測値の計算: モデルを使用してテキストの予測ラベルを取得
    with torch.no_grad():
        output = bert_mlm(**encoding)  # BERTモデルに符号化データを入力
        scores = output.logits  # モデルの出力からスコアを取得
        labels_predicted = scores[0].argmax(-1).cpu().numpy().tolist()  # 予測されたラベルを取得

    # ラベル列を文章に変換: 予測されたラベルを元のテキストに変換
    predict_text = tokenizer.convert_bert_output_to_text(
        text, labels_predicted, spans
    )

    return predict_text  # 予測されたテキストを返す

# いくつかの例に対してBERTによる文章校正を行ってみる。
text_list = [
    'ユーザーの試行に合わせた楽曲を配信する。',
    'メールに明日の会議の史料を添付した。',
    '乳酸菌で牛乳を発行するとヨーグルトができる。',
    '突然、子供が帰省を発した。'
]

# トークナイザ、ファインチューニング済みのモデルのロード
tokenizer = SC_tokenizer.from_pretrained(MODEL_NAME)  # SC_tokenizerを初期化
model = BertForMaskedLM_pl.load_from_checkpoint(best_model_path)  # 最良のモデルチェックポイントからモデルをロード
bert_mlm = model.bert_mlm.cuda()  # モデルをGPUに移動

for text in text_list:
    predict_text = predict(text, tokenizer, bert_mlm)  # BERTモデルによるテキストの修正予測
    print('---')
    print(f'入力：{text}')  # 入力されたテキストを表示
    print(f'出力：{predict_text}')  # 修正されたテキストを表示

# 9-17: BERTで予測を行い、正解数を数える。
correct_num = 0 
for sample in tqdm(dataset_test):
    wrong_text = sample['wrong_text']
    correct_text = sample['correct_text']
    predict_text = predict(wrong_text, tokenizer, bert_mlm) # BERT予測
   
    if correct_text == predict_text: # 正解の場合
        correct_num += 1

print(f'Accuracy: {correct_num/len(dataset_test):.2f}')

# 9-18: BERTモデルを用いて漢字の誤変換を特定し、精度を計算

correct_position_num = 0 # 正しく誤変換の漢字を特定できたデータの数
for sample in tqdm(dataset_test):  # テストデータセットをループ処理
    wrong_text = sample['wrong_text']  # 誤ったテキスト
    correct_text = sample['correct_text']  # 正しいテキスト
    
    # 符号化: 誤ったテキストの符号化
    encoding = tokenizer(wrong_text)
    wrong_input_ids = encoding['input_ids']  # 誤変換の文の符号列
    encoding = {k: torch.tensor([v]).cuda() for k, v in encoding.items()}  # 符号化データをGPUに転送
    correct_encoding = tokenizer(correct_text)  # 正しいテキストの符号化
    correct_input_ids = correct_encoding['input_ids']  # 正しい文の符号列
    
    # 文章を予測: モデルを使用して誤ったテキストの修正予測
    with torch.no_grad():
        output = bert_mlm(**encoding)  # BERTモデルに入力
        scores = output.logits  # 出力スコア（ロジット）
        predict_input_ids = scores[0].argmax(-1).cpu().numpy().tolist()  # 予測された文章のトークンID

    # 特殊トークンを取り除く
    wrong_input_ids = wrong_input_ids[1:-1]  # 誤ったテキストのトークンIDから特殊トークンを除外
    correct_input_ids = correct_input_ids[1:-1]  # 正しいテキストのトークンIDから特殊トークンを除外
    predict_input_ids = predict_input_ids[1:-1]  # 予測テキストのトークンIDから特殊トークンを除外
    
    # 誤変換した漢字を特定できているかを判定
    # 符合列を比較する。
    detect_flag = True
    for wrong_token, correct_token, predict_token in zip(wrong_input_ids, correct_input_ids, predict_input_ids):
        if wrong_token == correct_token:  # 正しいトークンの場合
            if wrong_token != predict_token:  # 正しいトークンが誤って別のトークンに変換されている場合
                detect_flag = False
                break
        else:  # 誤変換のトークンの場合
            if wrong_token == predict_token:  # 誤変換のトークンがそのままの場合
                detect_flag = False
                break

    if detect_flag:  # 誤変換の漢字の位置を正しく特定できた場合
        correct_position_num += 1  # 正しく特定できた数をカウントアップ
        
# 精度の計算と表示
print(f'Accuracy: {correct_position_num/len(dataset_test):.2f}')  # 精度（正確さ）を計算して表示
