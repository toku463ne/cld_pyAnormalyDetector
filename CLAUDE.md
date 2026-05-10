# CLAUDE.md — pyAnomalyDetector（再設計版）

## プロジェクト概要

ZabbixのDB（PostgreSQL / MySQL）からhistory/trendsデータを取得し、直近の異常を検出してZabbix DashboardやWeb UIに報告するツール。1時間に1回実行される。

### リファクタリングの目標

旧実装の3つの課題を解決する:
1. **ノイズ（誤検知）が多い** → 季節性を考慮した統計モデルと連続スコアリングで解決
2. **グルーピング精度が低い** → 2段階DBSCANを維持しつつパラメータを評価ベースで調整
3. **評価手段がない** → 評価フレームワーク（precision/recall）を最初から組み込む

### 旧実装の場所

`org/pyAnomalyDetector/` に旧コードが保存されている。参照用のみ。新実装はルート直下に構築する。

---

## リソース制約（最重要）

**本番Zabbixサーバ上で動作するため、計算・メモリリソースを最小化する。**

### 軽量化の指針

| 禁止・非推奨 | 代替手段 |
|---|---|
| 実行時にモデルをfit（sklearn等） | 統計量を事前計算してDBに保存し、検出時はDB参照のみ |
| STL分解（時系列全体をメモリに展開） | 時間帯別統計（hour-of-day stats）をDBに事前保存 |
| Isolation Forest（デフォルト使用） | オプション扱い。明示的に有効化した場合のみ実行 |
| 全itemのhistoryを一括メモリロード | バッチ処理（`batch_size`で分割）を必ず使う |
| NumPy/pandasの全列演算より複雑な操作 | GroupBy + 集計関数で済む処理はそれで完結させる |

### 事前計算の原則

検出実行時（毎時）に「重い計算」をしない。重い計算は事前バッチ（1〜2回/日）で行い、結果をDBに保存する:

- `trends_stats`テーブル: 過去N日のmean/stdを日次で更新
- `hour_stats`テーブル（新設）: 時間帯別のmean/stdを日次で更新 → SeasonalDetectorはこれを参照
- `history_stats`テーブル: 直近historyの統計量を毎時更新

検出時の1itemあたりの処理コストは「DBからの統計量取得 + 数回の四則演算」に収める。

---

## アーキテクチャ原則

1. **Protocolベースの抽象化**: ABCより`typing.Protocol`を優先。差し替え可能な実装を作りやすくする
2. **連続スコアリング**: 各Detectorは0.0〜1.0の異常スコアを返す。バイナリ判定はEnsemble後にのみ行う
3. **Functional core / Imperative shell**: 検出ロジックは純粋関数。副作用（DB書き込み等）はPipelineに集約
4. **Pydantic v2 by config**: 設定はPydanticモデルで型検証。`Dict[str, Any]`は使わない
5. **評価ファーストの開発**: 新しいDetectorを追加する前に、評価スクリプトでbefore/afterを比較する
6. **ログは構造化**: `logging`モジュール + `extra={}`で構造化ログを出力。`print()`は禁止
7. **事前計算・参照のみ**: 毎時の検出実行時は統計量のDB参照と軽量な演算のみ。fit/trainは禁止

---


## 検出アルゴリズム（新設計）

### 基本思想：連続スコアリング

旧実装の「段階フィルタ（detect1通過→detect2...）」をやめ、各Detectorが独立して**異常スコア（0.0〜1.0）**を計算する。EnsembleDetectorがスコアを合成し、最終的な閾値判定を行う。

```python
@dataclass
class AnomalyScore:
    item_id: int
    score: float                        # 最終スコア（0.0〜1.0）
    is_anomaly: bool                    # score >= threshold
    detector_scores: dict[str, float]   # 各Detectorのスコア内訳
    features: dict[str, float]          # 解釈のための特徴量
```

### ZScoreDetector（旧detect1の改良版）

- **計算**: `z = |recent_mean - trend_mean| / trend_std`
- **スコア**: `min(z / (2 * lambda_threshold), 1.0)` で連続値に変換
- **改良点**: 旧実装のバグ（演算子優先順位）を修正、`trend_std == 0`・`trend_mean == 0`のガードを追加

### ChangepointDetector（旧detect2の改良版）

- **計算**: CUSUMアルゴリズムで突発的な変化点を検出
- **スコア**: CUSUMスタットの最大値を正規化
- **改良点**: 旧実装の差分比較より変化の「持続性」を評価できる

### SeasonalDetector（旧detect3/4の代替）

- **事前計算**（日次バッチ）: 過去N日のhistory/trendsから「時間帯別 mean/std」を`hour_stats`テーブルに保存
  - キー: `(itemid, hour_of_day)` → `mean`, `std`（将来的に曜日別も可）
- **検出時（毎時）**: `hour_stats`から当該時間帯の期待値を1回DB参照し、直近meanとのz-scoreを計算
- **スコア**: `min(z / (2 * lambda_threshold), 1.0)`
- **計算量**: O(1)／item（STL不使用。事前計算済みの統計量を参照するだけ）
- **改良点**: 「毎日同じ時間帯に上昇するメトリクス」を誤検知しない

### EnsembleDetector

```yaml
ensemble:
  detectors:
    zscore:
      weight: 0.3
      lambda_threshold: 3.0
    changepoint:
      weight: 0.3
      cusum_h: 5.0
    seasonal:
      weight: 0.4
      lambda_threshold: 3.0
  min_score: 0.5        # これ以上でis_anomaly=True
  require_any: 1        # 最低1つのDetectorがcontributeすること
```

---

## 評価フレームワーク

### ラベルの定義

```python
class AnomalyLabel(Enum):
    NORMAL = 0
    ANOMALY = 1
    UNKNOWN = -1    # 評価から除外

@dataclass
class LabeledItem:
    item_id: int
    label: AnomalyLabel
    note: str = ""  # 異常の理由（任意）
```

### 評価データセット

2種類を用意する:

1. **合成データ**（`evaluation/synthetic.py`）:
   - 正常な時系列を生成（トレンド + 季節性 + ノイズ）
   - 既知のパターン（スパイク、シフト、ドリフト）を注入
   - ラベルは自動付与 → 単体テストに使用

2. **実データ**（`testdata/csv/20250508/`等）:
   - `testdata/labels/<dataset>/labels.csv` にラベルファイルを配置
   - フォーマット: `item_id,label,note`（label: 0=normal, 1=anomaly）
   - 統合テスト・パラメータチューニングに使用

### メトリクス

```python
@dataclass
class EvaluationReport:
    precision: float
    recall: float
    f1: float
    per_detector: dict[str, dict]   # 各Detectorの単独精度
    n_true_positive: int
    n_false_positive: int
    n_false_negative: int
    threshold_used: float
```

### バックテストの実行

```bash
# 合成データで評価
python -m evaluation.backtester --synthetic

# 実データで評価（ラベルファイル必要）
python -m evaluation.backtester \
  --dataset testdata/csv/20250508/psql \
  --labels testdata/labels/20250508_psql/labels.csv

# パラメータスイープ
python -m evaluation.backtester \
  --sweep \
  --output results/sweep_$(date +%Y%m%d).csv
```

---

## 設定スキーマ（新設計）

旧実装のYAML + `cascade_config()` は廃止し、Pydanticで型検証する。

```yaml
# config.yml（新形式）

admdb:
  host: "{{ ADM_DB_HOST }}"
  port: 5432
  user: anomdec
  password: "{{ ADM_DB_PASSWORD }}"
  dbname: anomdec

data_sources:
  production:
    type: zabbix_psql
    host: "{{ ZABBIX_DB_HOST }}"
    port: 5432
    user: zabbix
    password: "{{ ZABBIX_DB_PASSWORD }}"
    dbname: zabbix

    history_interval: 600       # 秒
    history_retention: 18       # 直近18ステップ保持
    trends_interval: 3600       # 秒
    trends_retention: 336       # 14日分（336時間）

    detectors:
      zscore:
        enabled: true
        weight: 0.3
        lambda_threshold: 3.0
        min_ignore_rate: 0.05
      changepoint:
        enabled: true
        weight: 0.3
        cusum_h: 5.0
        cusum_k: 0.5
      seasonal:
        enabled: true
        weight: 0.4
        period: 24
        robust: true

    ensemble:
      min_score: 0.5
      require_any: 1

    clustering:
      jaccard_eps: 0.1
      corr_eps: 0.2
      min_samples: 2
      sigma: 2.0
```

### 設定の上書き順序

1. `default.yml`（ベース値）
2. ユーザー指定の`config.yml`（上書き）
3. `ANOMDEC_SECRET_PATH` 環境変数で指定したsecretファイル（認証情報）
4. 環境変数（Jinja2テンプレートとして展開）

Pydanticの`model_validator`でdata_source内へのデフォルト値伝搬を行う。旧実装の`cascade_config()`は廃止。

---

## データフロー

```
[Zabbix DB / CSV]
      ↓ ingestion.DataSource
[生データ: (itemid, clock, value)]
      ↓ features.RollingStats / features.Seasonal
[特徴量: (mean, std, residual, seasonal, trend)]
      ↓ detectors.ZScore / Changepoint / Seasonal（並列実行）
[AnomalyScore per item per detector]
      ↓ detectors.Ensemble
[最終 AnomalyScore per item]
      ↓ clustering.DBSCAN
[クラスタリング済み異常リスト]
      ↓ store.Anomalies
[anomaliesテーブル]
      ↓ views.Flask / views.Streamlit
[UI/Dashboard]
```

---

## 管理DB（anomdec PostgreSQL）

テーブル命名規則: `{ds_name}_{table}`

| テーブル | 内容 | 更新タイミング |
|---|---|---|
| `{ds}_history` | 直近historyキャッシュ（itemid, clock, value） | 毎時 |
| `{ds}_history_stats` | historyの統計量（sum, sqr_sum, cnt, mean, std） | 毎時 |
| `{ds}_history_updates` | history更新範囲の記録 | 毎時 |
| `{ds}_trends_stats` | trendsの統計量（mean, std, cnt） | 日次 |
| `{ds}_trends_updates` | trends更新範囲の記録 | 日次 |
| `{ds}_hour_stats` | 時間帯別統計（itemid, hour_of_day → mean, std） | 日次（新設） |
| `{ds}_anomalies` | 検出異常（AnomalyScore + clusterid + detector_scores JSONB） | 毎時 |

旧実装からの変更点:
- `{ds}_hour_stats` テーブルを新設（SeasonalDetectorが検出時に参照）
- `anomalies`テーブルに`score`（FLOAT）と`detector_scores`（JSONB）カラムを追加

---

## テスト戦略

### 単体テスト（`tests/unit/`）

- 合成データを使う。DBへの依存なし
- 各Detectorを**純粋関数として**テスト（入力DataFrame → AnomalyScoreリスト）
- カバレッジ対象: `detectors/`, `features/`, `evaluation/metrics.py`, `config/schema.py`

```bash
python -m pytest tests/unit/ -v
```

### 統合テスト（`tests/integration/`）

- `testdata/csv/20250508/` の実データを使う
- PostgreSQL（`anomdec_test`）が必要
- 接続情報は `tests/test_secret.yml` に記載

```bash
python -m pytest tests/integration/ -v
```

---

## CLIエントリポイント

| スクリプト | 役割 | 実行頻度 | 重い処理 |
|---|---|---|---|
| `detect_anomalies.py` | 異常検出メインフロー | 毎時 | なし（DB参照と軽量演算のみ） |
| `update_stats.py` | trends_stats・hour_stats更新 | 1〜2回/日 | あり（夜間バッチ推奨） |
| `python -m evaluation.backtester` | オフライン評価 | 手動 | あり（開発環境推奨） |

```bash
# 異常検出
python detect_anomalies.py -c config.yml --end $(date +%s)

# 統計更新
python update_stats.py -c config.yml

# 評価
python -m evaluation.backtester \
  --dataset testdata/csv/20250508/psql \
  --labels testdata/labels/20250508_psql/labels.csv \
  --output results/eval_$(date +%Y%m%d).json
```

---

## 開発ガイドライン

### 新しいDetectorを追加するとき

1. `detectors/base.py` の `Detector` Protocolを実装する
2. `tests/unit/test_{name}_detector.py` を合成データで書く
3. `evaluation/backtester.py` で実データに対してbefore/afterを比較する
4. F1スコアが現状を下回らないことを確認してからmerge

### 禁止事項

- `Dict[str, Any]` をデータの受け渡しに使う（Pydanticモデルを使うこと）
- `print()` を使う（`logging` モジュールを使うこと）
- 単体テストでDBに接続する（`tests/unit/`はDBフリー）
- 副作用のある処理を `detectors/` に書く（`pipeline/` に書くこと）
- `detect_anomalies.py` の実行パス（毎時）でsklearnの`fit()`や統計量の全量再計算を行う
- 全itemのデータを一括でメモリに展開する（必ず`batch_size`で分割すること）

### 型注釈

Python 3.10+ の型記法を使う:

```python
# Good
def detect(items: list[int]) -> list[AnomalyScore]: ...
def get_data(item_id: int) -> pd.DataFrame | None: ...

# Bad
from typing import List, Optional
def detect(items: List[int]) -> List[AnomalyScore]: ...
```

---

## データソース

| タイプ | クラス | 説明 |
|---|---|---|
| `zabbix_psql` | `ZabbixPsqlSource` | Zabbix PostgreSQL DB |
| `zabbix_mysql` | `ZabbixMysqlSource` | Zabbix MySQL DB |
| `csv` | `CsvSource` | テスト用CSVファイル |
| `logan` | `LoganSource` | goLogAnalyzerのログデータ |

---

## 環境セットアップ

```bash
git clone https://github.com/toku463ne/pyAnomalyDetector.git
cd pyAnomalyDetector
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

---

## 用語集

| 用語 | 説明 |
|---|---|
| itemId | Zabbixの監視item（メトリクス）のID |
| history | Zabbixが収集した生データ（通常10分間隔） |
| trends | Zabbixが1時間ごとに集計したデータ（min/avg/max） |
| trends_stats | 過去N日分のtrends統計量（mean/std）。正常基準として使用 |
| epoch (ep) | Unix時間（秒） |
| clusterid | 同じ原因による異常グループのID（-1はノイズ/未分類） |
| data_source | データ取得元の設定単位（1つのZabbix DBが1つのdata_source） |
| admdb | 異常検出ツール自身の管理用PostgreSQL DB |
| AnomalyScore | Detectorが返す異常スコア（0.0〜1.0） |
