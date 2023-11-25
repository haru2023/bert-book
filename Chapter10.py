# 10.文章ベクトルを用いたデータの可視化と類似文章検索
# 実行方法："chap10"フォルダを作成し、"https://www.rondhuit.com/download/ldcc-20140209.tar.gz"を解凍して、VSCodeでChapter10.pyを実行

# カレントディレクトリを "./chap10/" にする
import os
file_directory = os.path.dirname(os.path.abspath(__file__)) # fileのあるディレクトリのパスを取得
target_directory = os.path.join(file_directory, 'chap10') # './chap10/'へのパスを構築
os.chdir(target_directory) # カレントディレクトリを変更

# 10-1: 'chap10'という名前の新しいフォルダを作成し、カレントディレクトリをそのフォルダに変更
#// !mkdir chap10
#// %cd ./chap10

# 10-2: 必要なライブラリをインストール
#// !pip install transformers==4.18.0 fugashi==1.1.0 ipadic==1.0.0

# 10-3: 必要なライブラリをインポート
import random
import glob
from tqdm import tqdm
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer, BertModel

# BERTの日本語モデル
MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'

# 10-4
#データのダウンロード
#// !wget https://www.rondhuit.com/download/ldcc-20140209.tar.gz 
#ファイルの解凍
#// !tar -zxf ldcc-20140209.tar.gz 

# 10-5: 必要なライブラリと変数の定義
category_list = [
    'dokujo-tsushin',   # カテゴリリストの定義
    'it-life-hack',
    'kaden-channel',
    'livedoor-homme',
    'movie-enter',
    'peachy',
    'smax',
    'sports-watch',
    'topic-news'
]

# トークナイザとモデルのロード
tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)  # 日本語用BERTトークナイザのロード
model = BertModel.from_pretrained(MODEL_NAME)                  # BERTモデルのロード
model = model.cuda()  # モデルをGPUに移動

# 各データの形式を整える
max_length = 256  # 最大トークン長の定義
sentence_vectors = [] # 文章ベクトルを追加していくリスト
labels = [] # ラベルを追加していくリスト
for label, category in enumerate(tqdm(category_list)):  # 各カテゴリに対してループ
    for file in glob.glob(f'./text/{category}/{category}*'):  # カテゴリ内のファイルに対してループ
        # 記事から文章を抜き出し、符号化を行う。
        with open(file, encoding='utf-8') as f:  # ファイルを開いて読み込む
            lines = f.read().splitlines()
        text = '\n'.join(lines[3:])  # 記事の本文を結合
        encoding = tokenizer(
            text, 
            max_length=max_length, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )  # トークナイザでテキストを符号化
        encoding = { k: v.cuda() for k, v in encoding.items() }  # 符号化されたデータをGPUに移動
        attention_mask = encoding['attention_mask']  # アテンションマスクの取得

        # 文章ベクトルを計算
        with torch.no_grad():  # 勾配計算を無効化
            output = model(**encoding)  # モデルによる出力の計算
            last_hidden_state = output.last_hidden_state 
            averaged_hidden_state = \
                (last_hidden_state*attention_mask.unsqueeze(-1)).sum(1) \
                / attention_mask.sum(1, keepdim=True)  # 最終層の出力の平均を計算

        # 文章ベクトルとラベルを追加
        sentence_vectors.append(averaged_hidden_state[0].cpu().numpy())  # 文章ベクトルをリストに追加
        labels.append(label)  # ラベルをリストに追加

# それぞれをnumpy.ndarrayにする。
sentence_vectors = np.vstack(sentence_vectors)  # 文章ベクトルのリストをnumpy配列に変換
labels = np.array(labels)  # ラベルのリストをnumpy配列に変換

# 10-6: 文章ベクトルのPCAによる次元削減
sentence_vectors_pca = PCA(n_components=2).fit_transform(sentence_vectors) # 文章ベクトルを2次元に削減
print(sentence_vectors_pca.shape) # 削減後の形状を出力

# 10-7: PCAによる次元削減の結果を可視化
plt.figure(figsize=(10,10))  # 描画領域のサイズを指定
for label in range(9):
    plt.subplot(3,3,label+1)  # 3x3のグリッドでサブプロットを作成
    index = labels == label  # 特定のラベルに対応するインデックスを取得
    plt.plot(
        sentence_vectors_pca[:,0], 
        sentence_vectors_pca[:,1], 
        'o', 
        markersize=1, 
        color=[0.7, 0.7, 0.7]  # 全てのデータポイントを灰色でプロット
    )
    plt.plot(
        sentence_vectors_pca[index,0], 
        sentence_vectors_pca[index,1], 
        'o', 
        markersize=2, 
        color='k'  # 特定のラベルに属するデータポイントを黒色でプロット
    )
    plt.title(category_list[label])  # サブプロットのタイトルをカテゴリ名で設定

# 10-8: 文章ベクトルのt-SNEによる次元削減
sentence_vectors_tsne = TSNE(n_components=2).fit_transform(sentence_vectors)  # 文章ベクトルをt-SNEを用いて2次元に削減

# 10-9: t-SNEによる次元削減の結果を可視化
plt.figure(figsize=(10,10))  # 描画領域のサイズを指定
for label in range(9):
    plt.subplot(3,3,label+1)  # 3x3のグリッドでサブプロットを作成
    index = labels == label  # 特定のラベルに対応するインデックスを取得
    plt.plot(
        sentence_vectors_tsne[:,0],
        sentence_vectors_tsne[:,1], 
        'o', 
        markersize=1, 
        color=[0.7, 0.7, 0.7]  # 全てのデータポイントを灰色でプロット
    )
    plt.plot(
        sentence_vectors_tsne[index,0],
        sentence_vectors_tsne[index,1], 
        'o',
        markersize=2,
        color='k'  # 特定のラベルに属するデータポイントを黒色でプロット
    )
    plt.title(category_list[label])  # サブプロットのタイトルをカテゴリ名で設定

# 10-10: 文書ベクトルの正規化と類似度計算
# 先にノルムを1にしておく。
norm = np.linalg.norm(sentence_vectors, axis=1, keepdims=True)  # 各文書ベクトルのノルムを計算
sentence_vectors_normalized = sentence_vectors / norm  # 文書ベクトルをノルムで割り、正規化

# 類似度行列の(i,j)要素はi番目の記事とj番目の記事の類似度を表している。
sim_matrix = sentence_vectors_normalized.dot(sentence_vectors_normalized.T)  # 正規化されたベクトル同士の内積により類似度行列を計算

# 入力と同じ記事が出力されることを避けるため、
# 類似度行列の対角要素の値を小さくしておく。
np.fill_diagonal(sim_matrix, -1)  # 自己対応を避けるために対角要素を-1に設定

# 類似度が高い記事のインデックスを得る
similar_news = sim_matrix.argmax(axis=1)  # 各文書に最も類似度が高い文書のインデックスを取得

# 類似文章検索により選ばれた記事とカテゴリーが同一であった記事の割合を計算
input_news_categories = labels  # 入力記事のカテゴリーラベル
output_news_categories = labels[similar_news]  # 類似記事のカテゴリーラベル
num_correct = (input_news_categories == output_news_categories).sum()  # カテゴリーが一致する記事の数を計算
accuracy = num_correct / labels.shape[0]  # 一致率（精度）を計算

print(f"Accuracy: {accuracy:.2f}")  # 精度を表示
