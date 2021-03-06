# README
## What is this?
This is a kedro usage report in [my blog](http://socinuit.hatenablog.com/entry/2020/02/08/210423).

個人ブログから転載し、一部をgithubでのmd記法に書き換えたものです。

# 個人的背景

仕事でも機械学習の案件がちょっと増えてきたというのと、  
kaggleもベースラインくらいは自動的にsubmitできるところまで持っていきたいって思ったので、  
pipelineを作ろうと言うことになりました。  

ただ、私はエンジニアリング畑ではないので、ゼロから作れる自信がありません。  
困ったなー困ったなーと思っていたところに、こんなQiitaを見かけました。

[PythonのPipelineパッケージ比較：Airflow, Luigi, Gokart, Metaflow, Kedro, PipelineX](https://qiita.com/Minyus86/items/70622a1502b92ac6b29c)

なるほど、いろいろあるんだな、となりました。  
この中から今回はKedroを導入しとりあえず触ってみたのでレポします。

<!-- more -->

## pipelineとは
機械学習のタスクをまとめ上げて、自動化するやつです(ざっくり)

データの前処理やパラメータのチューニングなど、機械学習の運用はいろんな煩雑化要素が現れて悲しくなります。  
このあたりを含めて整理し、結果の再現や

## Kedroとは

もうgithubに飛んだほうが早い  
https://github.com/quantumblacklabs/kedro

Quantumblack社が手がけるオープンソースのpipelineライブラリです。  

## なんでKedroなの？  

他のライブラリが悪い、というわけではなく、理由は主に3点です。

- ドキュメントが丁寧

- 開発に携わってる人が優しい

- ロゴ・バナーがかっこいい(重要)

### ドキュメントが丁寧
get startedはどのライブラリにもありますが、ステップバイステップで解説してくれており、  
ファイルの構造やいつも触る機会のないyamlファイルなどの対応関係がとてもわかりやすかったです。  
pipelineはおろか、Pythonも基本対話型で使っちゃうような私にとっては、この点はすごい重要でした。

### 開発者やコミュニティがニュービーに優しい
Twitterで「このやり方あとで調べよう」つったら  
[メインテインしてる人がリプくれたり](https://twitter.com/921kiyo/status/1225076745096830977)、  
[開発してる人から反応もらえたり](https://twitter.com/yetudada/status/1225071864394985473)、  
[悩んでいたコーディングの実装例を示してくれたり](https://twitter.com/Minyus86/status/1225203464910753792)して、  
びっくりした反面めちゃくちゃ嬉しかったというのがあります。  
多分他のライブラリを使ってても、誰かしら何かしら教えてくれるとは思うんですが、  
こうしていろいろ教えてくれるんだと思うだけで結構心強かったりします。

### ロゴ・バナーがかっこいい
このツイートが全てです。  

https://twitter.com/0_u0/status/1224950323757740032


## 使ってみた

ドキュメントがあるのでぶっちゃけここから先は蛇足でしかないんですが、使ってみたレポートです。  
実装githubはここです。

Kedroは`pip`でインストールできます。

```
pip install kedro
```

### 使ったデータ
[前ブログで使ったやつ](http://socinuit.hatenablog.com/entry/2019/12/19/132851)を改造したデータを使ってます。  
乱数で作ってるので、学習済みモデルも公開して問題ないっちゃ問題ない。何にも使えないので。  

### プロジェクト立ち上げは`kedro new`
Rを使っていれば割と自然なプロジェクトという概念、Kedroにもあります。  
コマンドラインに`kedro new`と入力すると対話的に「プロジェクト名」などを入力できます。  
これが済むと、作業ディレクトリ上にkedroのプロジェクトフォルダができます。しゅごい。

### 実行準備は`kedro install`
プロジェクトを立ち上げたら`kedro install`と入力するとなんかわにゃわにゃ動きます。  
動き終わると「データを置く場所」「前処理をする場所」「モデルを構築する場所」などが  
大体作られます。すごい。

### フォルダ構成
このgitではこんな感じになっています。  
`__init__`とか`__pycache__`とかもありますが省略しています。  
フォルダ構成の時点できれいなのも好きです。  
ただ、フォルダ量は膨大なので、各々の説明は公式ドキュメント等を参考にしてください。

```
kedro_classification
├── README.md
├── conf
│   ├── README.md
│   ├── base
│   │   ├── catalog.yml
│   │   ├── credentials.yml
│   │   ├── logging.yml
│   │   └── parameters.yml
│   └── local
├── data
│   ├── 01_raw
│   ├── 02_intermediate
│   ├── 03_primary
│   ├── 04_features
│   ├── 05_model_input
│   ├── 06_models
│   ├── 07_model_output
│   └── 08_reporting
├── docs
│   └── source
│       ├── conf.py
│       └── index.rst
├── kedro_cli.py
├── logs
│   ├── errors.log
│   ├── info.log
│   └── journals
├── notebooks
├── references
├── results
├── setup.cfg
└── src
    ├── kedro_classification
    │   ├── nodes
    │   ├── pipeline.py
    │   ├── pipelines
    │   │   ├── data_engineering
    │   │   │   ├── nodes.py
    │   │   │   └── pipeline.py
    │   │   └── data_science
    │   │       ├── nodes.py
    │   │       └── pipeline.py
    │   └── run.py
    ├── requirements.txt
    ├── setup.py
    └── tests
        └── test_run.py
```

### pipeline構築
上記のフォルダ構成ですが、第一に`data/01_raw`に使うデータを突っ込むというところをやります。  
今回はダミーデータがぶち込まれています。
それ以外では、`conf`にあるyamlファイルと、`src/[プロジェクト名]/pipelines/data_engineering`、  
同じく`data_science`のpythonスクリプト(`nodes.py`、`pipeline.py`)を編集していきます。
全部編集したら1階層上にも`pipeline.py`があるので、これも編集します。  
この辺の名前はおそらく参照さえしっかりしていればわかりやすく名前を付けても良いと思います。  

`conf`内のyamlファイルでは、中間テーブルの出力やモデルの出力を`catalog.yml`、  
モデルのパラメータなどを`parameters.yml`に格納します。  
これで、`nodes.py`で引数として指定すれば、Kedro側でよしなに引っ張ってきてくれます。
`nodes.py`には、具体的なデータの前処理やモデル関数の定義を書き、  
`pipeline.py`には、それらの入出力を定義します。

### data_engineering
このフォルダでは主に前処理を行います。  
今回はどちらかというとモデリングの部分のトライアルがメインだったので、  
適当な前処理になっちゃってます。

```python: data_engineering/nodes.py
import pandas as pd
import numpy as np


def preprocessing(usedata: pd.DataFrame) -> pd.DataFrame:
    for i in range(1,40):
        var_name = 'Var.' + str(i)
        usedata[var_name] = 1 - usedata[var_name]
    return usedata

```

もちろん複数関数を定義して、適宜適用することもできます。

そしてこれを、格納している元データに適用し、  
適用結果を中間テーブルとして吐き出すためのスクリプトが`pipeline.py`

```python: data_engineering/pipeline.py
from kedro.pipeline import node, Pipeline
from kedro_classification.pipelines.data_engineering.nodes import preprocessing

def create_pipeline(**kwargs):
    print('loading create_pipeline in pipeline.py....')
    return Pipeline(
        [
            node(
                func=preprocessing,
                inputs='usedata',
                outputs='preprocessed_Data',
                name='preprocessed_Data',
            ),
        ]
    )

```

実装も`nodes.py`の関数を持ってきて、  
出力は`preprocessed_Data`として出す、というシンプルなものです。  
元テーブル含め、使う中間テーブルは`catalog.yml`で管理されます。  
`pipeline.py`は`data_science`でも同様の記法で書きます。なので省略です。

### data_science
公式ドキュメントでは線型回帰の実装があったので、背伸びしてLightGBMの実装をやってみました。  
コード(一部)は以下。

なんかもしや？と思ったのですが、yamlファイルで管理しているパラメータを以下のようにすると、
「このモデルで使うパラメータはここ」という形で管理できます。
こうすることで`node.py`の中で管理しなくてもいいことになりました。

```yaml: parameters.yml

LightGBM_model: 
  n_estimators: 5000
  boosting_type: 'gbdt'
  objective: 'regression'
  metric: 'rmse'
  num_leaves: 50
  #subsample: 0.85
  #subsample_freq: 1
  learning_rate: 0.01
  #feature_fraction: 0.9
  max_depth: 5
  #lambda_l1: 0.1
  #lambda_l2: 0.1
  verbose: -1
  early_stopping_rounds: 1000
  #eval_metric: 'auc'


XGBoost_model: 
  num_rounds: 5000
  objective: 'binary:logistic'
  eval_metric: 'auc'
  booster: 'gbtree'
  n_jobs: 4
  tree_method: 'hist'
  eta: 0.01
  grow_policy: 'lossguide'
  max_delta_step: 2
  #seed: 538
  colsample_bylevel: 0.9
  colsample_bytree: 0.8
  gamma: 1.0
  learning_rate: 0.01
  max_bin: 64
  max_depth: 6
  max_leaves: 10
  min_child_weight: 10
  reg_alpha: 1e-06
  reg_lambda: 1.0
  subsample: 0.7
```

```python: data_science/nodes.py
def LightGBM_model(
    data: pd.DataFrame,
    #y: np.ndarray,
    parameters: Dict
    ) -> lgb.LGBMRegressor:
    
    ### define classes
    regressor = lgb.LGBMRegressor()
    y = data['y']
    X = data.drop(['y', 'ID'], axis=1)
    ### hyperparameters from parameters.yml
    lgb_params = parameters['LightGBM_model'] 

    ### fold?
    ### I want to define this function out of LightGBM_model.
    ### Keep thinking...
    fold = KFold(n_splits=parameters['folds'], random_state=parameters['random_state'])

    oof_pred = np.zeros(len(X))
    ### run model with kfold
    for k,   (train_index, valid_index) in enumerate(fold.split(X, y)):
        #print(train_index)
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_valid = lgb.Dataset(X_valid, y_valid)

        regressor = lgb.train(lgb_params, lgb_train, valid_sets=lgb_valid, verbose_eval=False)
        
        y_train_pred = regressor.predict(X_train, num_iteration=regressor.best_iteration)
        y_valid_pred = regressor.predict(X_valid, num_iteration=regressor.best_iteration)

        auc_train = roc_auc_score(y_train, y_train_pred)
        auc_valid = roc_auc_score(y_valid, y_valid_pred)
        print('Early stopping round is: {iter}'.format(iter=regressor.current_iteration()))
        print('Fold {n_folds}: train AUC is {train: .3f} valid AUC is {valid: .3f}'.format(n_folds=k+1, train=auc_train, valid=auc_valid))

        #train_roc = roc_auc_score()
    #print(type(regressor))
    
    ### todo: evaluation, predict oof...
    
    return regressor


def evaluate_LightGBM_model(regressor: lgb.basic.Booster, X_test: np.ndarray, y_test: np.ndarray): 
    y_pred = regressor.predict(X_test, num_iteration=regressor.best_iteration)
    print('y predicted!')
    print(type(y_pred)) 
    #y_pred = np.argmax(y_pred, axis=1)
    #roc_curve = r
    score  = roc_auc_score(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info('AUC is %.3f.', score)
```

ついでにXGBoostも同様の形で実装したのでメモ。

``` python: data_science/node.py
def XGBoost_model(
    data:pd.DataFrame,
    parameters:Dict
) -> xgb.core.Booster:

    fold_xgb = KFold(n_splits=parameters['folds'], random_state=parameters['random_state'])
    train_df = data.drop('ID', axis=1)
    xgb_params = parameters['XGBoost_model']
    num_rounds = xgb_params['num_rounds']
    y = train_df[parameters['target']]
    train_df = train_df.drop(parameters['target'], axis=1)
    feature = train_df.columns.values

    for fold_, (train_idx, valid_idx) in enumerate(fold_xgb.split(train_df.values)):
        train_x, train_y = train_df.iloc[train_idx], y.iloc[train_idx]
        valid_x, valid_y = train_df.iloc[valid_idx], y.iloc[valid_idx]
        print("fold n °{}".format(fold_+1))
        trn_Data = xgb.DMatrix(train_x, label = train_y, feature_names=feature)
        val_Data = xgb.DMatrix(valid_x, label = valid_y, feature_names=feature)
        watchlist = [(trn_Data, "Train"), (val_Data, "Valid")]
        print("xgb trainng folds " + str(fold_) + "-" * 50)
        xgb_model = xgb.train(xgb_params, trn_Data,num_rounds,watchlist,early_stopping_rounds=100, verbose_eval= 1000)
        del train_idx,valid_idx
        gc.collect()
    return xgb_model

def evaluate_XGBoost_model(regressor: xgb.core.Booster, X_test: np.ndarray, y_test: np.ndarray): 
    #X_test = X_test.values
    #print(regressor.feature_names)
    xgb_test = xgb.DMatrix(X_test, feature_names=regressor.feature_names)
    y_pred = regressor.predict(xgb_test, ntree_limit=regressor.best_ntree_limit)
    print("y predicted on XGBoost!")
    print(type(y_pred)) 
    #y_pred = np.argmax(y_pred, axis=1)
    #roc_curve = r
    score  = roc_auc_score(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("XGBoost AUC is %.3f.", score)
```

汚い実装で恥ずかしいですが、こんな感じで書ければ大丈夫です。  
`pipeline.py`でこれらの関数と入出力を指定すればOK。

```python: data_science/pipeline.py
# coding: utf-8

from kedro.pipeline import node, Pipeline
from typing import Dict, Any, List
from kedro_classification.pipelines.data_science.nodes import (
    split_data,
    Linear_Regression_model,
    LightGBM_model,
    evaluate_LightGBM_model
)

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=split_data,
                inputs=['preprocessed_Data', 'parameters'],
                outputs=['X_train', 'X_test', 'y_train', 'y_test'],
            ),            
            node(
                func=LightGBM_model,
                inputs=['preprocessed_Data', 'parameters'],
                outputs='regressor',
                name='regressor',
            ),
            node(
                func=evaluate_LightGBM_model,
                inputs=['regressor', 'X_test', 'y_test'],
                outputs=None,
            ),
        ]
    )

```

最後に、`src`直下の`pipeline.py`を編集します。  
ここで前処理からモデル実行・出力に至るまでの流れを指定します。

```python: src/pipeline.py
from typing import Dict

from kedro.pipeline import Pipeline
from kedro_classification.pipelines.data_engineering import pipeline as de
from kedro_classification.pipelines.data_science import pipeline as ds

def create_pipelines(**kwargs) -> Dict[str, Pipeline]:
    de_pipeline = de.create_pipeline()
    ds_pipeline = ds.create_pipeline()

    return {
        'de' : de_pipeline,
        '__default__': de_pipeline + ds_pipeline
    }
```

### `kedro test`
ここまでやったら(あるいはここまでに至るまでのどこかで)テストを実行できます。
コマンドラインで`kedro test`を実行するだけでどこで詰むか詰まないかがわかります。すごい。

### `kedro run`
テストが上手く行ったら走らせます。`kedro run`で走ります。  
すごい。

## おわりに
長くなりました。  
ここまでの実装は、ドキュメントを見ながら試行錯誤し、3日位で無理なく進められました。  
今回はローカルでの実装ですが、もちろんKedroはAWSやGCPでも動きますし、  
様々なDBからデータを獲得できます。
データは分析が主で、パイプライン構築以前にこういうエンジニアリングの経験が浅い私にも  
どうにか形にできるところまで道案内してくれるので、とても良いライブラリでした。

一回ベースラインができたとはいえ、まだまだ足りないことも多いです。  
現状はLighGBMのモデル関数の中でCVの手法をねじ込んでいる部分の改善や、  
複数の前処理関数を組合せたpipeline構築など、  
脱Tutorialな課題はいっぱいあります。  
ただ、これはKedro側の問題というよりは`nodes.py`内でどう実装するかの問題なので、  
Kedroは割と柔軟にここを受け入れてくれると信じています(？)  
