#使い方#

1. 以下のサイトからmodelをダウンロード・解凍し、facialexp_conv以下に配置（facialexp_conv/model/emo2Imgやtxt2emoにする）  
https://drive.google.com/open?id=1xCQE84BRIWyXZTk-YPVRznMWA1lA3ras
  
  https://drive.google.com/open?id=1BbyqZJtgSLAiPFsSEEPC5SJHr36BMkRB
  ※テキスト→感情の日本語用モデル追加

2. main.ipynbを実行

#FAQ#
- `warning: CRLF will be replaced by LF in src/DEEPCommunication/DEEPCommunication.py.
The file will have its original line endings in your working directory.`
と出て、勝手にファイルが書き換わったときはどうする？
    - .gitattiruteのせいなので、`vi .gitattribute`などで、`* text=auto`の前に`#`を付けてコメントアウトする
    - その後、`git checkout HEAD .gitattribute`としたら治ります
    - ref : https://qiita.com/shunsuke_takahashi/items/d02f8085451b9aa4ffcf
