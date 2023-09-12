# 在Windows上設置ThreatReportExtractor

這個README提供了在Windows系統上設置和運行ThreatReportExtractor專案的逐步指南。

## 環境設置

1. **安裝Python 3.8**: 確保你的系統已經安裝了Python 3.8，因為這個專案需要Python 3.8的虛擬環境。

2. **克隆GitHub倉庫**: 打開命令提示符並執行以下命令來克隆專案：
    ```bash
    git clone https://github.com/jackaduma/ThreatReportExtractor.git
    ```

3. **進入專案目錄**:
    ```bash
    cd ThreatReportExtractor
    ```

## 安裝依賴

4. **安裝Python依賴**:
    ```bash
    pip install -r requirements.txt
    ```

    或者，如果您使用Pipenv，您可以運行以下命令：
    ```bash
    pipenv install --python 3.8
    pipenv install -r requirements.txt
    ```

5. **安裝Spacy模型**:
    ```bash
    python -m spacy download en_core_web_lg
    ```

6. **安裝NLTK**:
    ```python
    import nltk
    nltk.download('averaged_perceptron_tagger')
    ```

7. **初始化和更新Git子模塊**:
    ```bash
    git submodule init
    git submodule update
    ```

8. **下載AllenNLP預訓練模型**: 打開PowerShell，然後執行：
    ```powershell
    Invoke-WebRequest -Uri "https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz" -OutFile "srl-model-2018.05.25.tar.gz"
    Rename-Item -Path "srl-model-2018.05.25.tar.gz" -NewName "srl-model.tar.gz"
    ```

9. **安裝Graphviz**: 從[Graphviz官方網站](https://graphviz.gitlab.io/_pages/Download/Download_windows.html)下載適合你系統的安裝包，然後按照安裝指示完成安裝。

## 特別注意

### 版本兼容性
確保你的spaCy和neuralcoref版本是兼容的。版本衝突是Python項目中常見的問題，特別是當使用多個庫或框架時。

### 模型下載
通過正確地下載`en_core_web_lg`模型，您解決了spaCy找不到該模型的問題。這是運行專案的關鍵一步。

## 運行專案

10. **運行主程序**:
    ```bash
    python3 main.py --asterisk true --crf true --rmdup true --elip true --input_file input.txt --gname mygraph
    ```
