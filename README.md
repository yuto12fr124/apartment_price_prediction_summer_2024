## 使用したハードウェア

### ローカル環境

- **CPU**: 11th Gen Intel Core i7-11370H (4コア/8スレッド)
- **GPU**: NVIDIA GeForce RTX 3050 Ti (4GB GDDR6)
- **メモリ**: 32GB LPDDR4x
- **ディスクサイズ**: 2TB SSD

### Google Colaboratory


## 使用したOS
### ローカル環境
- **OS**: Windows 11

## バージョン情報
エディション	Windows 11 Home
バージョン	23H2
インストール日	‎2022/‎10/‎06
OS ビルド	22631.4169
シリアル番号	0F010ME221300C
エクスペリエンス	Windows Feature Experience Pack 1000.22700.1034.0



## 乱数シードなどの設定値の情報
utils.pyやその他のコード内で必要になる度にハードコーディングで設定しています。全てが乱数シード(42)を設定しています。Catboostのみ乱数シードを設定していません。
utils_test.pyというコードもありますが、こちらはコンペ開催期間中に使用したOptunaの最適化のコードが含まれます。Optunaの最適化自体には乱数シードを設定していません。


## 使用するクラスや関数が格納されたコードについて
`project_directory/code/utils.py`に使用するクラスや関数が格納されており、ほぼ全てのコードで呼び出しています。(一部のコードは.ipynb内に直接書いています。)


## モデルの学習から予測まで行う際の、ソースコードの実行手順。
※提出したファイルの最上位のディレクトリ「project_directory」からつながるパスで記載しています。

### 1. 前処理

#### 実行するノートブック
* **ファイル名:** `preprocess.ipynb`
* **保存場所:** `project_directory/code`


**入力データ:**

* **コンペティションデータ:**
    * `data/competition_data/train/train/*.csv`: 学習用データ
    * `data/competition_data/test.csv`: テストデータ
* **外部データ:**
    * `data/external_data/station20240426free.csv`: 駅データ (駅データ.jp)
    * `data/external_data/pref.csv`: 都道府県対応表 (駅データ.jp)
* **中間結果:**
    * `data/intermediate_results/checkpoints/all_address.csv`: すべての住所の経度緯度情報 (国土地理院API)
    * `data/intermediate_results/checkpoints/address.csv`: 使用する住所の経度緯度情報 (国土地理院API)

**出力データ:**

* `data/intermediate_results/checkpoints/preprocess_main_table.csv`: 前処理後のメインデータ


### 2. モデルの学習と予測

#### 実行するノートブック
* **LightGBM(目的変数:単位面積あたりの取引価格_log)ファイル名:** `train_lightgbm_l1_cv.ipynb`
* **LightGBM(目的変数:取引価格(総額)_log)ファイル名:** `train_lightgbm_l1_cv_tg.ipynb`
* **Catboost(目的変数:単位面積あたりの取引価格_log)ファイル名:** `train_cat_lq_cv.ipynb`
* **SARIMAX(目的変数:四半期毎の取引価格(総額)_logの平均)ファイル名:** `train_arima.ipynb`
* **保存場所:** `project_directory/code`

**入力データ:**

* **前処理済みデータ:**
    * `project_directory/data/intermediate_results/checkpoints/preprocess_main_table.csv`: 前処理済みのデータ

* **テーブルデータの行に対する並び順を保つデータ:**
    ※作業環境(ローカル,コンペ開催期間中に使用したGoogle Driveのディレクトリ,入賞者提出物を作成するためのGoogle Driveのディレクトリ)によって`project_directory/data/competition_data/train`内のCSVファイルの並び順が異なることにより学習の再現結果が変動してしまいました。それぞれの環境のID順を再現するための.npyファイルになります。
    * `project_directory/data/intermediate_results/checkpoints/train_index_l1_cv.npy`: `train_lightgbm_l1_cv.ipynb`でdf_train対して使用
    * `project_directory/data/intermediate_results/checkpoints/test_index_l1_cv.npy`: `train_lightgbm_l1_cv.ipynb`でdf_testに対して使用
    * `project_directory/data/intermediate_results/checkpoints/train_index.npy`: `train_lightgbm_l1_cv_tg.ipynb`でdf_train対して使用
    * `project_directory/data/intermediate_results/checkpoints/test_index.npy`: `train_lightgbm_l1_cv_tg.ipynb`でdf_testに対して使用
    * `project_directory/data/intermediate_results/checkpoints/train_index_cat_lq_cv.npy`: `train_catboost_lq_cv.ipynb`でdf_train対して使用
    * `project_directory/data/intermediate_results/checkpoints/test_index_cat_lq_cv.npy`: `train_catboost_lq_cv.ipynb`でdf_testに対して使用

**出力データ:**
* **8Foldの交差検証で得られた学習データへの予測結果**
    * `project_directory/data/cv_predictions/sub_oof_lgbm_l1_cv`: `train_lightgbm_l1_cv.ipynb`で生成される
    * `project_directory/data/cv_predictions/sub_oof_lgbm_l1_cv_tg`: `train_lightgbm_l1_cv_tg.ipynb`で生成される
    * `project_directory/data/cv_predictions/sub_oof_cat_lq_cv`: `train_lightgbm_l1_cv.ipynb`で生成される

* **時系列モデルで得られた四半期の2期分の予測結果**
    * `project_directory/data/intermediate_results/checkpoints/pred_時系列mean_colabo.csv`: `train_arima.ipynb`で生成される

* **testデータに対する予測結果**
    * `project_directory/data/intermediate_results/one_model_predictions/sub_test_lgbm_l1_cv.csv`: `train_lightgbm_l1_cv.ipynb`
    * `project_directory/data/intermediate_results/one_model_predictions/sub_test_lgbm_l1_cv_tg.csv`: `train_lightgbm_l1_cv_tg.ipynb`
    * `project_directory/data/intermediate_results/one_model_predictions/sub_test_cat_lq_cv.csv`: `train_catboost_lq_cv.ipynb`


### 3. アンサンブルと最終予測

#### 実行するノートブック
* **ファイル名:**`ensemble_weights.ipynb`
* **保存場所:**`project_directory/code`

**入力データ:**

* **学習データに対する予測結果:**
    * `project_directory/data/cv_predictions/sub_oof_lgbm_df.csv`: `train_lightgbm_l1_cv.ipynb`から出力されたもの
    * `project_directory/data/cv_predictions/sub_oof_lgbm_target_df.csv`: `train_lightgbm_l1_cv_tg.ipynb`から出力されたもの
    * `project_directory/data/cv_predictions/sub_oof_cat_lq_cv.csv`: `train_cat_lq_cv.ipynb`から出力されたもの
    * `project_directory/data/intermediate_results/checkpoints/df_arima_mean.csv`:`train_arima.ipynb`から出力されたもの

* **testデータに対する予測結果**
    * `project_directory/data/intermediate_results/one_model_predictions/sub_test_lgbm_l1_cv.csv`
    * `project_directory/data/intermediate_results/one_model_predictions/sub_test_lgbm_l1_cv_tg.csv`
    * `project_directory/data/intermediate_results/one_model_predictions/sub_test_cat_lq_cv.csv`

**出力データ:**
* `project_directory/data/final_predictions/test_lgbm_lgbm_tg_cat_NMtime.csv`最終予測結果





## 学習済みモデルを使用して予測のみ行う場合の、ソースコードの実行手順

### 1. 前処理

#### 実行するノートブック
* **ファイル名:** `preprocess.ipynb`
* **保存場所:** `project_directory/code`


**入力データ:**

* **コンペティションデータ:**
    * `data/competition_data/train/train/*.csv`: 学習用データ
    * `data/competition_data/test.csv`: テストデータ
* **外部データ:**
    * `data/external_data/station20240426free.csv`: 駅データ (駅データ.jp)
    * `data/external_data/pref.csv`: 都道府県対応表 (駅データ.jp)
* **中間結果:**
    * `data/intermediate_results/checkpoints/all_address.csv`: すべての住所の経度緯度情報 (国土地理院API)
    * `data/intermediate_results/checkpoints/address.csv`: 使用する住所の経度緯度情報 (国土地理院API)

**出力データ:**

* `data/intermediate_results/checkpoints/preprocess_main_table.csv`: 前処理後のメインデータ

### 2. 学習済みモデルでの予測

#### 実行するノートブック
* **LightGBM(目的変数:単位面積あたりの取引価格_log)ファイル名:** `run_model_lightgbm_l1_cv.ipynb`
* **LightGBM(目的変数:取引価格(総額)_log)ファイル名:** `run_model_lightgbm_l1_cv_tg.ipynb`
* **Catboost(目的変数:単位面積あたりの取引価格_log)ファイル名:** `run_model_cat_lq_cv.ipynb`
* **SARIMAX(目的変数:四半期毎の取引価格(総額)_logの平均)ファイル名:** `run_model_sarimax.ipynb`
* **保存場所:** `project_directory/code`

**入力データ:**

* **前処理済みデータ:**
    * `project_directory/data/intermediate_results/checkpoints/preprocess_main_table.csv`: 前処理済みのデータ

* **学習済みモデル:**
    
    * `project_directory/models/model_lgbm_l1_cv_20240908_110735_0_06585289841.pkl`: `run_model_lightgbm_l1_cv.ipynb`で使用する学習済みモデル
    * `project_directory/models/model_lgbm_l1_cv_target_20240910_144521_0_06636916360273092.pkl`: `run_model_lightgbm_l1_cv_tg.ipynb`で使用する学習済みモデル
    * `project_directory/models/v1model_cat_lq_fold_1.cbm`~`project_directory/models/v1model_cat_lq_fold_8.cbm`: `run_model_cat_lq_cv.ipynb`で使用する学習済みモデル(fold1~8の合計8個があります)
    * `project_directory/models/model_sarimax.pkl`: `run_model_sarimax.ipynb`で使用する学習済みモデル

* **開催期間中に使用したCatboostモデル** 
    * `project_directory/models/model_cat_lq_cv_20240908_141325_0_06609571831620502.pkl`: 開催期間中に使用したCatboostモデル

**出力データ:**

* **時系列モデルで得られた四半期の2期分の予測結果**
    * `project_directory/data/intermediate_results/checkpoints/pred_時系列mean_colabo.csv`: `train_arima.ipynb`で生成される

* **testデータに対する予測結果**
    * `project_directory/data/intermediate_results/one_model_predictions/test_lgbm_cv.csv`: `train_lightgbm_l1_cv.ipynb`で生成される
    * `project_directory/data/intermediate_results/one_model_predictions/test_lgbm_cv_target.csv`: `train_lightgbm_l1_cv_tg.ipynb`で生成される
    * `project_directory/data/intermediate_results/one_model_predictions/test_cat_cv.csv`: `train_catboost_lq_cv.ipynb`で生成される

### 3. アンサンブルと最終予測

#### 実行するノートブック
* **ファイル名:**`ensemble_weights.ipynb`
* **保存場所:**`project_directory/code`

**入力データ:**

* **学習データに対する予測結果:**
    * `project_directory/data/cv_predictions/oof_lgbm_df.csv`: `train_lightgbm_l1_cv.ipynb`から出力されたもの
    * `project_directory/data/cv_predictions/oof_lgbm_target_df.csv`: `train_lightgbm_l1_cv_tg.ipynb`から出力されたもの
    * `project_directory/data/cv_predictions/oof_cat_df.csv`: `train_cat_lq_cv.ipynb`から出力されたもの
    * `project_directory/data/intermediate_results/checkpoints/df_arima_mean.csv`:`train_arima.ipynb`から出力されたもの

* **testデータに対する予測結果**
    * `project_directory/data/final_predictions/test_lgbm_cv20240908_111634.csv`: 開催期間中に「LightGBM(目的変数:単位面積あたりの取引価格_log)」で予測したもの
    * `project_directory/data/final_predictions/test_l1_tg20240910_144521.csv`: 開催期間中に「LightGBM(目的変数:取引価格(総額)_log)」で予測したもの
    * `project_directory/data/final_predictions/v1test_cat_cv20240919_111048.csv`: 開催期間中の「Catboost(目的変数:単位面積あたりの取引価格_log)」を再現して予測したもの

* **開催期間中に使用したCatboostモデルの予測結果**
    * `project_directory/data/final_predictions/test_cat_cv20240908_141325修正.csv` 

**出力データ:**
* `project_directory/data/final_predictions/test_lgbm_lgbm_tg_cat_NMtime.csv`最終予測結果

## Catboostモデルについて
- コンペ開催期間中は`model_cat_lq_cv_20240908_141325_0_06609571831620502.pkl`から出力された予測結果を使用していたのですが、現在は破損しているかメモリが不足していて読み込めません。
可能な限り同じものを再現したCatboostモデルを`.cbm`の保存形式で残しています。

## Optunaについて
- `project_directory/code/utils_test.py``project_directory/code/optuna_holdout.ipynb`を使用してパラメータ最適化を試せるようにしてあります。こちらは乱数シードを設定しておらず、学習データの順番も完全には再現できていないのでどのような解になるかは不確定要素があります。
- コンペ開催期間中は1モデル辺り40時間以上(大体50~100回)程度の最適化を行い、最も精度が良かったパラメータをハードコーディングで交差検証のパラメータとして与えています。
- `project_directory/code/optuna_holdout.ipynb`はコード内のコメントアウトしてあるコードを必要に応じて#を付けたり消したりする事で実行しています。もしくはコードを複製して同時に実行します。

## コードが実行時に前提とする条件
- コードの全てを絶対パスで書いてしまっているので、ディレクトリ構成に合わせて`project_directory`より上の階層のコードを修正する必要があります。
- Optunaでパラメータの最適化を20時間以上行う場合、enqueue_trialを使用してもっと精度が良かったパラメータを引き継がせます。(Google Colaboの24時間制限があるため)
- run_model_sarimax.ipynbはpandas,numpyを旧バージョンしなければ動作しません。

## コードの重要な副作用
- 保存操作が行われるコードはファイル名が同じ場合は既存のファイルを上書きします。(コンペ開催期間中に使用したモデルや予測結果,1位のスコアの再現をしたモデルや予測結果が含まれます)
- 決定木系の3モデルが生成する学習済みモデルのサイズが1Fold->3GBに達することがあるので空き容量に注意してください。

## 出力ファイルの命名規則について
- コンペ開催期間終了後~9/24(火)の期間中に手元で動作や再現性を確認するために使用したコードの出力ファイルはファイル名の先頭に`sub`をつけてあります。(※異なる名前がついたファイルもあるので提出までに可能な限り修正します。申し訳ございません。)

## その他のコードについて
- `project_directory/code/geocode_api.ipynb`: 国土地理院APIからの経度緯度等の情報を取得するコードです(コンペ開催期間中に取得したデータは1ヶ月以上前のものなので全く同じデータが得られない可能もあります。)
