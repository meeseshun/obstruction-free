# 発表資料
https://docs.google.com/presentation/d/1-guijUu2h1J5_FAIR-rjPVBAAFifuNQ2cbODJx-BDZ4/edit?usp=sharing

# 実行方法

`./predict.py -vv [--save-debug-images] [-i] [-gpu] BASE_IMAGE IMAGES [IMAGES ...] -o OUTPUT_IMAGE`

# 注意
プログラムのカレントディレクトリで実行してください。

--save-debug-images オプションでデバック画像を保存します。

-gpu オプションは対応していればGPUを計算に使用します。

前景が重なった画像が出力される場合は -i オプションを指定してください。

# 使用したプログラム

https://github.com/OLIET2357/pytorch-siggraph2017-inpainting
