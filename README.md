# yt_pyXspec

## Introduction

support pyXspec on jupyter with matplotlib

## Install

1. clone this repository to local: `git clone https://github.com/TomoTom0/yt_pyXspec.git`
2. in the local directory, local install with pip in established mode: `pip install -e .`

## Usage

For example

```python
import os, glob2, re
import yt_pyXspec

# work directory path
dir_path=os.environ["HOME"]+"/XXXXXXXXXXX/fit"

# how to define a graph title from the xcm path
# if title_func or its return value is None, there is no title
def title_func(xcm_path):
    return os.path.split(xcm_path)[1]

# how to define a image path from the xcm path
# if exportImagePath_func or its return value is None, there is no export file
def exportImagePath_func(xcm_path):
    return os.path.split(xcm_path)[1].replace(".xcm", ".pdf")

# automatically chage directory to dir_path
ytpx=yt_pyXspec.ytpx(dir_path)

# select xcm paths whose plot you want to export as image file
# in the appropriate method
xcm_paths=["AAAA.xcm", "BBBB.xcm"]

# plot graph from xcm paths and export as image file
# plots allows [eeu, eem, ld, del, ratio]
fig_axs_dic=ytpx.plot_xyss_fromXcms(
    xcms=xcm_paths,
    title_func=title_func,
    exportImagePath_func=exportImagePath_func,
    plots=["eeu", "del"])
```

## Contact

email: tomoIris427+Github@gamil.com

## LICENSE

MIT