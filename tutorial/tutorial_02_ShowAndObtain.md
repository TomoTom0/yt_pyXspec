# yt_pyXspecを利用した効率化 Step2 - Show and Obtain -

「yt_pyXspecを利用した効率化 Step1」に引き続き、yt_pyXspecの利用に関するTutorialを続ける。
公式のpyXspecのドキュメントも要参考。「pyXspec XXX」の検索で簡単にアクセスできる。

## 情報の表示・取得

|関数|内容|
|-|-|
|`ytpx.showParamsAll()`|`show parameter`および`show fit`相当の結果を`texttable`で整形して表示する|
|`ytpx.obtainInfoParamsAll()`|`show parameter`および`show fit`相当の結果を辞書形式で取得する|
|`ytpx.showParamsAll_xspec()`|pyXspecで`show parameter`および`show fit`相当の操作を行う|
|`ytpx.obtainInfoParamsForExport()`|`show parameter`および`show fit`相当の結果をスプレッドシートなどへのエクスポートに適したtsv形式で表示およびクリップボードにコピーする|

### ytpx.showParamsAll()

基本的にはxspecの`show parameter`および`show fit`相当の結果を表示している。差異は以下の通り。
- pyXspecではデータセット内のパラメータ番号がよく用いられることを鑑み、通しのパラメータ番号`i_pT`とデータセット内のパラメータ番号`i_p`を用意した。
- スペース節約のため、`index`を`i`、`parameter`を`p`または`param`、`component`を`c`または`comp`、`model`を`m`と省略している。
    - また、`i_m`はpyXspecとしてはモデル番号だが、Xspecとしてはデータセット番号。
- `elf`は`error`ないし`link`、`frozen`の情報を表示する欄 (Xspecでは単に`error`)。
    - ここでのエラーはfit (`xspec.Fit.perform()`)実行時に得られるエラーではなく、`error` (`xspec.Fit.error()`)で得られるもの (デフォルトでは1 sigma相当)。
    - `error`未実行の場合、下限上限ともに0.0が与えられる。なお、表においては`<下限>_<上限>`で表記される。
- 引数で`flag_oneOnly=True`を与えれば、1番目のデータセットの結果のみ表示される。(2番目以降のデータセットが1番目の値とすべてリンクされている場合に便利)
- 引数の`order_value`(既定値は5)に応じて、表中の数値の最大有効桁数が設定される。
    - `format(val, ".5g")`のような関数で文字列に変換しており、指数表示と小数表示は自動で調整される。

```python
ytpx.showParamsAll()
"""
i_p   i_pT   i_c   i_m   name_comp   name_param   unit      value     elf_0   elf_1
===================================================================================
                                                                                   
  1      1     1     1   phabs       nH           10^22   0.27539     0       0    
  2      2     2     1   powerlaw    PhoIndex             2.6366      0       0    
  3      3     2     1   powerlaw    norm                 0.0020709   0       0    
  4      4     3     1   diskbb      Tin          keV     3.4262      0       0    
  5      5     3     1   diskbb      norm                 0.0027117   0       0    
                                                                                   
  1      6     1     2   phabs       nH           10^22   0.27539     = p1    null 
  2      7     2     2   powerlaw    PhoIndex             2.6366      = p2    null 
  3      8     2     2   powerlaw    norm                 0.0020709   = p3    null 
  4      9     3     2   diskbb      Tin          keV     3.4262      = p4    null 
  5     10     3     2   diskbb      norm                 0.0027117   = p5    null 
                                                                                   
  1     11     1     3   phabs       nH           10^22   0.27539     = p1    null 
  2     12     2     3   powerlaw    PhoIndex             2.6366      = p2    null 
  3     13     2     3   powerlaw    norm                 0.0020709   = p3    null 
  4     14     3     3   diskbb      Tin          keV     3.4262      = p4    null 
  5     15     3     3   diskbb      norm                 0.0027117   = p5    null 
                                                                                   
  1     16     1     4   phabs       nH           10^22   0.27539     = p1    null 
  2     17     2     4   powerlaw    PhoIndex             2.6366      = p2    null 
  3     18     2     4   powerlaw    norm                 0.0020709   = p3    null 
  4     19     3     4   diskbb      Tin          keV     3.4262      = p4    null 
  5     20     3     4   diskbb      norm                 0.0027117   = p5    null 
		Reduced Chi-Squared: 2238.2 / 1654 = 1.3532
"""

# ytpx.Fit.perform()
# ytpx.Fit.error("1 2 3 4 5")

ytpx.showParamsAll(flag_oneOnly=True)
"""
i_p   i_pT   i_c   i_m   name_comp   name_param   unit      value       elf_0       elf_1  
===========================================================================================
                                                                                           
  1      1     1     1   phabs       nH           10^22   0.27566     0.25721     0.29659  
  2      2     2     1   powerlaw    PhoIndex             2.6387      2.5038      2.7905   
  3      3     2     1   powerlaw    norm                 0.0020725   0.0019607   0.0022061
  4      4     3     1   diskbb      Tin          keV     3.4261      3.363       3.4913   
  5      5     3     1   diskbb      norm                 0.0027147   0.0024454   0.003024 
		Reduced Chi-Squared: 2238.2 / 1654 = 1.3532
"""
```

### ytpx.obtainInfoParamsAll()

- `ytpx.showParamsAll()`で表示している内容が辞書形式で与えられる。
- 引数`flag_forPrint`の`True`/`False`によって返り値に含まれる情報が若干調整される。

### ytpx.showParamsAll_xspec()

- pyXspecの機能を利用して`show parameter`および`show fit`相当の結果を表示する。
- 具体的には`xspec.AllModels.show()`と`xspec.Fit.show()`を走らせている。
- 引数で`flag_oneOnly=True`を与えれば、1番目のデータセットの結果のみ表示される。
    - この場合は`xspec.AllModels(1).show()`と`xspec.Fit.show()`を走らせる。

### ytpx.obtainInfoParamsForExport()

- `ytpx.showParamsAll()`で表示している内容がtsv形式(行方向がタブ`\t`連結、列方向が改行`\n`連結)で与えられる。
    - xcmファイルのpath、reduced chisqも追加行に出力される。
- オプションに応じてprint (`flag_print`)、クリップボードへのコピー(`flag_copy`)の有効無効を切り替えられる。
- 出力をそのままスプレッドシートなどに貼り付けることで、表が得られる。

## xspecにおけるmodel, component, parameter

- model: `phabs * (powerlaw + diskbb)`
- component: `phabs`, `powerlaw`, `diskbb`

xspecではcomponentをmodelと称することもあるので若干ややこしいが、**単体のものがcomponent**、それらを **組み合わせたものがmodel**である。
parameterはcomponentの結果を調整するパラメータである(その名の通り)。

### pyXspecでの取り扱い

- `model = xspec.AllModels(ind_dataGroup)`とすれば`ind_dataGroup`(データグループのindex、1始まり)に対応するmodelオブジェクトが得られる。
- componentの名前は`model.componentNames`でlistとして取得できる。
- componentオブジェクト自体はmodelのattributeとして取得可能。
    - 文字列としてcomponent名がある場合、`obj.__dict__`attributeを利用して`model.__dict__[componentName_str]`とする。
- parameterとcomponentの関係はcomponentとmodelの関係と同じ。
    - parameterの名前は`comp.parameterNames`でlistとして取得可能。
    - parameterオブジェクト自体はcomponentのattributeとして取得可能。
    - `comp.__dict__[parameterName_str]`でも取得できる。
- parameterのattributeとして値やfrozenなども取得できる。詳細はpyXspecのドキュメントを参考に。

```python
model = xspec.AllModels(1)

print(model.expression)
# 'phabs(powerlaw + diskbb)'

componentNames = model.conponentNames
# ['phabs', 'powerlaw', 'diskbb']

comps = [model.__dict__[compName] for compName in model.componentNames]

comp = comps[0]
print(comp.name)
# 'phabs'

params = [comp.__dict__[paramName] for paramName in comp.parameterNames]
param = params[0]
print(param.name)
# 'nH'
print(param.values)
# [0.2756618071046863, 0.002756618071046863, 0.0, 0.0, 100000.0, 1000000.0]
```

## Fitting

当然ながら、pyXspecでもFitting関連の操作ができる。
xspecでの`fit`は`xspec.Fit.perform()`、`error 1 2 3`は`xspec.Fit.error("1 2 3")`で実行できる。

