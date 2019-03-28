## 使い方

1. モデルファイルダウンロード・解凍
facialexp_conv以下に配置（facialexp_conv/model/emo2Imgやtxt2emoにする）  
- 
　https://drive.google.com/open?id=1xCQE84BRIWyXZTk-YPVRznMWA1lA3ras
  
  https://drive.google.com/open?id=1BbyqZJtgSLAiPFsSEEPC5SJHr36BMkRB
  ※テキスト→感情の日本語用モデル追加

2. 下記フォルダ構成に格納する。
```
facialexp_conv
├── inp
│   └── production ※フォルダを作成してください
├── models
│   ├── cv2
│   │   └── haarcascade_frontalface_default.xml ※OpenCV用モデルファイル
│   ├── emo2Img ※フォルダを作成してください
│   │   ├── 200000-D.ckpt ※StarGAN 64x64用モデルファイル
│   │   └── 200000-G.ckpt ※StarGAN 64x64用モデルファイル
│   ├── emo2Img256 ※フォルダを作成してください
│   │   ├── 200000-D.ckpt ※StarGAN 256x256用モデルファイル
│   │   └── 200000-G.ckpt ※StarGAN 256x256用モデルファイル
│   └── txt2emo ※フォルダを作成してください
│       ├── checkpoint-0.923.h5       ※テキスト→感情モデルファイル
│       ├── model_2018-08-28-15_00.h5 ※テキスト→感情モデルファイル
│       └── tokenizer_cnn_ja.pkl      ※テキスト→感情モデルファイル
├── res  ※フォルダを作成してください
└── src
    ├── DEEPCommunication
    │   └── fasttext_model_30.bin ※fasttext用モデルファイル
    ├── stg
    └── tec
```

3. 必要なモジュールインストール

4. main.ipynbを実行

## FAQ
- `warning: CRLF will be replaced by LF in src/DEEPCommunication/DEEPCommunication.py.
The file will have its original line endings in your working directory.`
と出て、勝手にファイルが書き換わったときはどうする？
    - .gitattiruteのせいなので、`vi .gitattribute`などで、`* text=auto`の前に`#`を付けてコメントアウトする
    - その後、`git checkout HEAD .gitattribute`としたら治ります
    - ref : https://qiita.com/shunsuke_takahashi/items/d02f8085451b9aa4ffcf
- CUIでmodelsをダウンロードするには？
    - `$ FILE_ID=1xCQE84BRIWyXZTk-YPVRznMWA1lA3ras`
    - `$ FILE_NAME=models.zip`
    - `$ curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /dev/null`
    - `$ CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"`
    - `$ curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${FILE_ID}" -o ${FILE_NAME}`
    - とすると、1個目のmodelsはダウンロードできる
    - もう一つは  `FILE_ID=1BbyqZJtgSLAiPFsSEEPC5SJHr36BMkRB` として上のcurl以下を繰り返すとダウンロードできる
- CUIでのzip解凍は？
    - `unar models.zip`がおすすめ
    - `unar`がない場合、`sudo apt install unar`
        - ilectの場合は`sudo`は付けない
