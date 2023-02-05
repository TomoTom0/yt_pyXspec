# yt_pyXspec

## Introduction

support pyXspec on jupyter with matplotlib

## Install

1. clone this repository to local: `git clone https://github.com/TomoTom0/yt_pyXspec.git`
2. move to the cloned directory, local install with `pip` in the established mode at the same hierarchy as `setup.py`: `pip install -e .`

## Usage

For example,

```python
import os, glob2, re
import yt_pyXspec

# working directory path
dir_path=os.environ["HOME"]+"/XXXXXXXXXXX/fit"

# how to define a graph title from the xcm path
# if title_func or its return value is None, there is no title
def title_func(xcm_path):
    #return (["double_pow"+s for s in re.findall(r"\d{4}-\d{2}-\d{2}", os.path.split(xcm_path)[1])]+[None])[0]
    return os.path.split(xcm_path)[1]

# how to define a image path from the xcm path
# if exportImagePath_func or its return value is None, there is no export file
def exportImagePath_func(xcm_path):
    #return "graph_eeu-del_"+(re.findall(r"\d{4}-\d{2}-\d{2}", os.path.split(xcm_path)[1])+[None])[0]+".pdf"
    return os.path.split(xcm_path)[1].replace(".xcm", ".pdf")

# automatically chage directory to dir_path
ytpx=yt_pyXspec.ytpx(dir_path)

# select xcm paths whose plot you want to export as image file
# in the appropriate method; glob2.glob("all_doublePow*2022-04-*.xcm")
xcm_paths=["AAAA.xcm", "BBBB.xcm"]

# plot graph from xcm paths and export as image file
# plots allow "eeu", "eem", "ld", "del", "ratio"

fig_axs_dic=ytpx.plot_datas_fromXcms(
    xcms=xcm_paths,
    title_func=title_func,
    exportImagePath_func=exportImagePath_func,
    plots=["eeu", "del"])
```

### arguments for plot_datass

#### plots
- iterable; `["eeu", "ld"]`
- plot graphs of the inputed type
- allowed values are `"eeu", "eem", "ld", "del", "ratio"`

#### x_lim
- iterable; `[range_min, range_max]`
- a range of x-axis
- if not inputted, the range is determined by the value of data

#### y_lims
- dict; `{"eeu": [range_min, range_max], "ld":[range_min, range_max]}`
- ranges of y-axis for plot types
- if not inputted, the range is determined by the value of data

#### colors
- iterable; `["royalblue", "red", "oliverdrab"]`
- a color set of plot
- check the reference of matplotlib to know valid values

#### markers
- iterable; `["^", "o", "+"]`
- a marker set of plot
- check the reference of matplotlib.pyplot.scatter to know valid values

#### datas_s
- dict `{"eeu": datas}`
- if not inputted, datas_s are obtained from the present pyXspec environments
- if you use `plot_datas_fromXcms`, you should not input `datas_s`

#### exportImagePath
- str `"/home/XXXXXXXXXXX/AAA.pdf"`
- if not inputed, a image of the graph is not exported
- check the reference of matplotlib.pyplot.figure.savefig to know allowed extensions
- if you use `plot_datas_fromXcms`, you should use `exportImagePath_func`

#### title
- str `"AAAAAAA"`
- a title on the top of the graph
- if not inputed, there is no title
- if you use `plot_datas_fromXcms`, you should use `title_func`

#### __others

- legends_dic
- legends_sort

- flag_dataPlot: True
- flag_modelPlot: True
- flag_compPlot: True
- flag_modelStep: False
- marker_size_data: 0
- elinewith_data: 1
- marker_size_mdoel: 0
- elinewidth_model: 0.5

## Contact

email: tomoIris427+Github@gmail.com

## LICENSE

MIT