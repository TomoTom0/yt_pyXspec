# import
import os
import re
import sys
import shutil

# import glob2
import texttable
import warnings
import pyperclip
import xspec
import numpy as np
import pandas as pd
import scipy
import astropy
import subprocess
import multiprocessing
from IPython.display import clear_output
import time
import pickle
from pathlib import Path  # , Int, List, Dict, Str
from typing import Optional, Union

import matplotlib.pyplot as _plt
from matplotlib import gridspec


def mjd2date(
    mjd: Union[float, str],
    format_in: str = "mjd",
    format_out: str = "iso",
    subfmt_out: Optional[str] = "date",
) -> str:
    return (astropy.time.Time([str(mjd)], format=format_in)).to_value(
        format_out, subfmt=subfmt_out
    )[0]


def date2mjd(
    date_str: str,
    format_in: str = "iso",
    format_out: str = "mjd",
    subfmt_out: Optional[str] = None,
) -> float:
    return (astropy.time.Time([date_str], format=format_in)).to_value(
        format_out, subfmt=subfmt_out
    )[0]


# from https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def clearCell(seconds=0):
    if is_notebook() is True:
        time.sleep(seconds)
        clear_output()


def romanNumeral(num):
    return chr(0x215F + num)


def injectSafeChar_forPath(phrase):
    pattern = r"[^0-9a-zA-Z\-_]"
    return re.sub(pattern, "-", phrase)


def convertUnit(vals, unit_from, unit_to):
    return [
        (s * astropy.units.__dict__[unit_from])
        .to(astropy.units.__dict__[unit_to], equivalencies=astropy.units.spectral())
        .value
        for s in vals
    ]


class Ytpx:
    # # init
    def __init__(
        self,
        dir_path: Optional[Union[Path, str]] = None,
        _sys: Optional[any] = None,
        flag_parallel: bool = True,
        flag_ipythonClear: bool = True,
        flag_cacheData: bool = True,
    ) -> None:
        self.dir_path: Optional[Union[Path, str]] = None
        self.chDir(dir_path=dir_path)
        self.findXcm(dir_path=dir_path)

        self.Plot: xspec.Plot = xspec.Plot
        self.AllModels: xspec.AllModels = xspec.AllModels
        self.AllData: xspec.AllData = xspec.AllData
        self.Xset: xspec.Xset = xspec.Xset
        self.Fit: xspec.Fit = xspec.Fit
        self.plt: _plt = _plt
        self.xspec: xspec = xspec
        self.flag_ipythonClear: bool = flag_ipythonClear
        self._sys: sys = _sys if _sys is not None else sys
        _sys = self._sys
        self.flag_cacheData: bool = flag_cacheData

        self.__stdout__: sys.stdout = _sys.stdout
        self.__stderr__: sys.stderr = _sys.stderr
        self.default_exportImagePath: Optional[Path] = None
        self.xcm: Optional[list[Path]] = None
        self.initMatplotlibRcParams()
        self.initXspecPlot()
        if flag_parallel is True:
            self.set_parallel()
        # self.gridspec=_gridspec

    def chDir(self, dir_path: Optional[Union[Path, str]] = None) -> None:
        dir_path: Path = Path(dir_path)
        if dir_path.is_dir():
            print(f"change directory to {dir_path}")
            os.chdir(dir_path)
            self.dir_path = dir_path

    def findXcm(
        self, keyword: Optional[str] = None, dir_path: Optional[Union[Path, str]] = None
    ) -> list[Path]:
        valid_dir_path: Path = Path(dir_path or self.dir_path)
        valid_keyword: str = keyword or "*.xcm"
        self.xcm_keyword: str = valid_keyword
        xcms = sorted([s for s in valid_dir_path.glob(valid_keyword)])
        self.xcms = xcms
        return xcms

    def set_parallel(self, num_para: Optional[int] = None) -> None:
        num_para: int = num_para or multiprocessing.cpu_count() * 2
        self.Xset.parallel.leven: int = num_para
        self.Xset.parallel.error: int = num_para

    def codeXcm(self, xcm_path: Optional[Union[Path, str]] = None) -> None:
        xcm_path: Path = Path(xcm_path or self.xcm)
        subprocess.run(["code", xcm_path])

    # Enable or Disable Print
    def _managePrint(self, enable=True) -> None:
        _sys = self._sys
        if enable is True:
            _sys.stdout: sys.stdout = self.__stdout__
            _sys.stderr: sys.stderr = self.__stderr__
        else:
            _sys.stdout: sys.stdout = open(os.devnull, "w")
            _sys.stderr: sys.stderr = open(os.devnull, "w")

    def initXspecPlot(self) -> None:
        Plot: xspec.Plot = self.Plot
        Plot.xAxis: str = "keV"
        Plot.addCommand("time off")
        Plot.addCommand("font roman")

        Plot.addCommand("cs 2")
        Plot.addCommand("lw 5")
        Plot.addCommand("win 1")
        Plot.addCommand("loc 0.05 0.05 1 0.95")
        Plot.addCommand("win 2")
        Plot.addCommand("loc 0.05 0.05 1 0.95")

    def initMatplotlibRcParams(self, flag_forSlide: bool = False) -> None:
        plt: _plt = self.plt
        plt.rcParams["figure.figsize"] = (3.14 * 2, 3.14 * 2 * 0.7)
        plt.rcParams["figure.dpi"] = 100  # 画像保存するときは300に
        plt.rcParams["savefig.dpi"] = 300  # 画像保存するときは300に

        factor: float = 1.8 if flag_forSlide is True else 1
        factor_size: float = 1.2 if flag_forSlide is True else 1

        plt.rcParams["font.family"] = "Times New Roman"  # 全体のフォントを設定
        plt.rcParams["mathtext.fontset"] = "stix"  # 数式のフォントを設定
        plt.rcParams["xtick.direction"] = "in"  # x軸の目盛線を内向きへ
        plt.rcParams["ytick.direction"] = "in"  # y軸の目盛線を内向きへ
        plt.rcParams["xtick.minor.visible"] = True  # x軸補助目盛りの追加
        plt.rcParams["ytick.minor.visible"] = True  # y軸補助目盛りの追加
        plt.rcParams["xtick.major.width"] = 1.5 * factor  # x軸主目盛り線の線幅
        plt.rcParams["ytick.major.width"] = 1.5 * factor  # y軸主目盛り線の線幅
        plt.rcParams["xtick.minor.width"] = 1.0 * factor  # x軸補助目盛り線の線幅
        plt.rcParams["ytick.minor.width"] = 1.0 * factor  # y軸補助目盛り線の線幅
        plt.rcParams["xtick.major.size"] = 10 * factor_size  # x軸主目盛り線の長さ
        plt.rcParams["ytick.major.size"] = 10 * factor_size  # y軸主目盛り線の長さ
        plt.rcParams["xtick.minor.size"] = 5 * factor_size  # x軸補助目盛り線の長さ
        plt.rcParams["ytick.minor.size"] = 5 * factor_size  # y軸補助目盛り線の長さ
        plt.rcParams["font.size"] = 14 * factor  # フォントの大きさ
        plt.rcParams["axes.linewidth"] = 1.5 * factor  # 囲みの太さ
        plt.rcParams["axes.labelsize"] = "large" if flag_forSlide is True else "medium"
        plt.rcParams["xtick.labelsize"] = (
            "medium" if flag_forSlide is True else "medium"
        )
        plt.rcParams["ytick.labelsize"] = (
            "medium" if flag_forSlide is True else "medium"
        )

    # # load

    def _set_verboses(self, verboses: list[int, int] = [0, 0]):
        verboses_old: list = []
        for verbose, chatter in zip(
            verboses, [self.Xset.chatter, self.Xset.logChatter]
        ):
            verboses_old.append(chatter)
            if isinstance(verbose, int):
                chatter = verbose
        return verboses

    def loadXcm(self, xcm_path: Union[Path, str], verbose: int = 0):
        xcm_path = Path(xcm_path)
        if not xcm_path.is_file():
            return False
        if isinstance(verbose, int):
            verboses_old = self._set_verboses([verbose, verbose])
        self.Xset.restore(str(xcm_path))
        self._set_verboses(verboses_old)
        self.xcm: Path = xcm_path
        self.setRebinAll()
        if self.flag_ipythonClear is True:
            clearCell(3)
        return True

    def saveXcm(self, xcm_path: Union[Path, str], overwrite=True):
        xcm_path = Path(xcm_path)
        if xcm_path.is_dir():
            print(f"{xcm_path} is directory")
            return False
        elif xcm_path.is_file() and overwrite is True:
            tmp_name: Path = xcm_path + ".copy"
            shutil.copy(xcm_path, tmp_name)
            try:
                os.remove(xcm_path)
                self.Xset.save(xcm_path)
            except Exception as e:
                os.rename(tmp_name, xcm_path)
        else:
            self.Xset.save(str(xcm_path))

    def readFits(self, fits_paths):
        fits_str = " ".join(
            [f"{ind+1}:{ind+1} {s}" for ind, s in enumerate(fits_paths)]
        )
        self.AllData(fits_str)

    def setRebinAll(self, minSig=3, maxBins=80):
        stateMethod = self.Fit.statMethod
        for num_data in range(1, self.AllData.nSpectra + 1):
            # print(self.AllData(num_data)._Spectrum__fileName)
            if stateMethod in ["chi"]:
                self.Plot.setRebin(1, 1, num_data)
            elif stateMethod in ["cstat"]:
                self.Plot.setRebin(minSig, maxBins, num_data)

    def obtain_fitsNames(self):
        return [
            self.AllData(num_data)._Spectrum__fileName
            for num_data in range(1, self.AllData.nSpectra + 1)
        ]

    # # obtain_datass
    def obtain_datass_s_fromXcms(self, xcms=[], **kwargs_in):
        # plots
        datass_s_dic = {}
        for xcm_path in xcms:
            self.loadXcm(xcm_path)
            datass_now = self.obtain_datass(**kwargs_in)
            datass_s_dic[xcm_path] = datass_now
        return datass_s_dic

    def _obtainAdditiveComps(self, model):
        comps = [model.__dict__[s] for s in model.componentNames]
        return [comp for comp in comps if "norm" in comp.parameterNames]

    def obtain_datass(self, plots=["eeu"]):
        Plot = self.Plot
        AllData = self.AllData
        AllModels = self.AllModels
        plot_command = " ".join(plots)
        Plot(plot_command)
        datass = {}
        for tmp_plotWindow, tmp_plotType in enumerate(plots):
            plotType_orig = self.obtainOriginalPlotType(tmp_plotType, flag_strict=True)
            if plotType_orig is None:
                raise Exception(f"{tmp_plotType} may be invalid plotType")
            # self._managePrint(False)
            plotType = plotType_orig
            plotWindow = tmp_plotWindow + 1
            datas = {}
            datas_info = {
                "labels": {
                    key_xy: re.sub(r"\$([^\$]*)\$", r"$\\mathdefault{\1}$", label)
                    for key_xy, label in zip(["x", "y"], Plot.labels(plotWindow))
                },
                "title": Plot.labels(plotWindow)[2],
                "log": {
                    key_xy: _log
                    for key_xy, _log in zip(["x", "y"], [Plot.xLog, Plot.yLog])
                },
                "model": AllModels(1).expression,
                "plotWindow": plotWindow,
                "xcm_fileName": self.xcm,
            }

            # xs, ys, xe, ye, ys_model, ys_comps = [[]]*6
            for plotGroup in range(1, AllData.nGroups + 1):
                dataFuncs_dict = {
                    "xs": Plot.x,
                    "ys": Plot.y,
                    "xe": Plot.xErr,
                    "ye": Plot.yErr,
                    "ys_model": Plot.model,
                }
                datas_data = {}
                for key_data, dataFunc in dataFuncs_dict.items():
                    try:
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore")
                            data_obtained = dataFunc(plotGroup, plotWindow)
                        # warnings.resetwarnings()
                        datas_data[key_data] = data_obtained
                    except Exception as e:
                        pass

                model = AllModels(plotGroup)
                datas_groupInfo = {"src_fileName": AllData(plotGroup).fileName}

                # obtain comps in models
                comps_obtained = []
                compNames = []
                addComps = self._obtainAdditiveComps(model)
                datas_comp = {}
                if len(addComps) > 1:
                    for ind_compAdd, compAdd in enumerate(addComps):
                        try:
                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore")
                                comp_tmp = Plot.addComp(
                                    ind_compAdd + 1, plotGroup, plotWindow
                                )
                            # execlude components with only 0
                            if all(s == 0 for s in comp_tmp):
                                continue
                            comps_obtained.append(comp_tmp)
                            compNames.append(compAdd.name)
                        except Exception as e:
                            break
                    if len(compNames) > 0:
                        datas_comp = {
                            "ys_comps": comps_obtained,
                            "compNames": compNames,
                        }
                datas[plotGroup] = {**datas_groupInfo, **datas_data, **datas_comp}
            datass[plotType] = {"info": datas_info, "data": datas}
            # self._managePrint(True)
        if self.flag_ipythonClear is True:
            clearCell(2)

        return datass

    def obtain_datass_dic(self, xcms_dic, plots_dic):
        Xset = self.Xset
        datass_dic = {}
        for key_xcms, xcm_paths in xcms_dic.items():
            datass_s = {}
            for key_xcm, xcm_path in xcm_paths.items():
                self.loadXcm(xcm_path)
                datass_s[key_xcm] = self.obtain_datass(plots=plots_dic[key_xcms])

            datass_dic[key_xcms] = self.combine_datass_s(datass_s=datass_s)
        return datass_dic

    def combine_datass_s(self, datass_s={}, sortKeys=[]):
        datass_new = {}
        dataIndexs = {}
        if set(sortKeys) == set(datass_s.keys()):
            datass_s_list = [datass_s[k] for k in sortKeys]
        else:
            datass_s_list = list(datass_s.values())
        for datass in datass_s_list:
            for plotName, datas_dicts in datass.items():
                if plotName not in datass_new.keys():
                    dataIndexs[plotName] = 1
                    datass_new[plotName] = {}
                for datas_dict in datas_dicts.values():
                    datass_new[plotName][dataIndexs[plotName]] = datas_dict
                    dataIndexs[plotName] += 1
        return datass_new

    # # sub functions for plot
    def _checkWritableFilePath(self, path):
        if path is None:
            return False
        isFile_OK = os.path.isfile(path) and os.access(path, os.W_OK)
        isNotExist_OK = os.path.isdir(os.path.split(path)[0]) and not os.path.exists(
            path
        )
        return isFile_OK or isNotExist_OK

    def _extractValues(self, datas):
        sum_dic = {}
        for key_xy in ["x", "y"]:
            valss = [s.get(key_xy + "s", []) for s in datas.values()]
            errss = [s.get(key_xy + "e", []) for s in datas.values()]
            models = [s.get(key_xy + "s_model", []) for s in datas.values()]
            sum_tmp = []
            for vals, errs in zip(valss, errss):
                if len(errs) == len(vals):
                    sum_tmp += list(
                        np.array(vals * 2)
                        + np.array(errs * 2)
                        * np.array([1] * len(errs) + [-1] * len(errs))
                    )
                else:
                    sum_tmp += vals
            sum_dic[key_xy + "s"] = sum_tmp + sum(models, [])
            # xs_sum=sum([np.array(s["xs"]*2)+np.array(s["xe"]*2)*np.array([1]*len(s["xe"])+[-1]*len(s["xe"])) for s in data["eeu"].values()], [])
            # ys_sum=sum([np.array(s["ys"]*2)+np.array(s["ye"]*2)*np.array([1]*len(s["ye"])+[-1]*len(s["ye"])) for s in data["eeu"].values()], [])
        return sum_dic  # {"xs":xs_sum, "ys":ys_sum}

    def _obtainLim(self, values, logIsValid=True, margin_ratio=0.05):
        valid_values = [s for s in values if (not logIsValid) or s > 0]
        if len(valid_values) == 0:
            return []
        v_min = min(valid_values)
        v_max = max(valid_values)
        if logIsValid is True:
            margin = (np.log10(v_max) - np.log10(v_min)) * margin_ratio
            return (
                10 ** (np.log10(np.array([v_min, v_max])) + np.array([-1, +1]) * margin)
            ).tolist()
        else:
            margin = (v_max - v_min) * 0.05
            return ((np.array([v_min, v_max])) + np.array([-1, +1]) * margin).tolist()

    def _categorizePlotType(self, plotType, categoryType="log"):
        categorize_dict = {
            "log": [
                "lcounts",
                "ldata",
                "ufspec",
                "eufspec",
                "eeufspec",
                "model",
                "emodel",
                "eemodel",
            ],
            "big": [
                "lcounts",
                "ldata",
                "ufspec",
                "eufspec",
                "eeufspec",
                "model",
                "emodel",
                "eemodel",
                "counts",
                "data",
                "background",
                "chain",
                "contour",
                "dem",
                "eqw",
            ],
        }
        return any(
            key.startswith(plotType) for key in categorize_dict.get(categoryType, [])
        )

    def obtainOriginalPlotType(self, plotType, flag_strict=True):
        original_plotTypes = [
            "goodness",
            "lcounts",
            "eemodel",
            "delchi",
            "margin",
            "chain",
            "polangle",
            "icounts",
            "eeufspec",
            "ratio",
            "chisq",
            "eqw",
            "dem",
            "counts",
            "fitstat",
            "efficien",
            "polfrac",
            "residuals",
            "ufspec",
            "data",
            "insensitv",
            "eufspec",
            "contour",
            "integprob",
            "emodel",
            "model",
            "ldata",
            "sensitvity",
            "background",
            "sum",
        ]
        tmp_plotType_origs = [s for s in original_plotTypes if s.startswith(plotType)]
        if flag_strict is not True:
            return tmp_plotType_origs[0]
        elif len(tmp_plotType_origs) == 1:
            return tmp_plotType_origs[0]
        else:
            return None

    def _dictGet(self, dic, keys, default_value=None):
        if not isinstance(dic, dict):
            return default_value
        return ([dic[key] for key in keys if key in dic.keys()] + [default_value])[0]

    def _judgeReal(self, value):
        return isinstance(value, int) or isinstance(value, float)

    def set_limsFromDatas(
        self,
        datas,
        ax=None,
        x_lim=None,
        y_lim=None,
        scaleIsLog=None,
        flag_inclusive=False,
    ):
        # set lim
        lims_in = {"x": x_lim, "y": y_lim}
        lims = {}
        data_sum = self._extractValues(datas)
        if scaleIsLog is None:
            if ax is None:
                scaleIsLog = {"x": True, "y": True}
            else:
                scaleIsLog = {
                    "x": ax.get_xscale() == "log",
                    "y": ax.get_yscale() == "log",
                }
        lims_fromValue = {
            key_xy: self._obtainLim(
                data_sum[key_xy + "s"], logIsValid=scaleIsLog.get(key_xy, True)
            )
            for key_xy in ["x", "y"]
        }
        self.lims_fromValue = lims_fromValue
        if len(lims_fromValue["y"]) == 0 and "ys_model" in data_sum.keys():
            lims_fromValue["y"] = self._obtainLim(
                data_sum["ys_model"], logIsValid=scaleIsLog.get("y", True)
            )

        lims = lims_fromValue
        # y_lim = self._dictGet(y_lims, [plotType_input, plotType_orig])
        for key_xy in ["x", "y"]:
            lim_in = lims_in[key_xy]
            lim_data = lims[key_xy]
            if hasattr(lim_in, "__iter__") and not len(lim_in) < 2:
                if flag_inclusive is True:
                    lims[key_xy] = [
                        min(lim_in[0], lim_data[0]),
                        max(lim_in[1], lim_data[1]),
                    ]
                else:
                    lims[key_xy] = lim_in
        if ax is not None:
            ax.set_xlim(lims["x"])
            ax.set_ylim(lims["y"])

        return lims

    # # plot datass

    def plot_datass(
        self,
        plots=["eeufspec"],
        x_lim=[],
        y_lims={},
        colors=[
            "royalblue",
            "red",
            "olivedrab",
            "turquoise",
            "orange",
            "chartreuse",
            "navy",
            "firebrick",
            "darkgreen",
            "darkmagenta",
        ],
        markers=[],
        # legends_dic={},
        # legends_sort=[],
        datass=None,
        exportImagePath=None,
        **kwargs_in,
    ):
        # kwargs
        kwargs_default = {
            "flag_dataPlot": True,
            "flag_modelPlot": True,
            "flag_compPlot": True,
            "flag_modelStep": False,
            "marker_size_data": 0,
            "elinewith_data": 1,
            "marker_size_mdoel": 0,
            "elinewidth_model": 0.5,
            "title": None,
            "xlabel": None,
            "ylabel_dic": None,
            "ylabel_fontsize_dic": None,
            "facecolor": "white",
            "subplots_in": None,
            "compParams_abs": [],
        }
        kwargs = {**kwargs_default, **kwargs_in}

        plt = self.plt
        default_exportImagePath = self.default_exportImagePath
        obtain_datass = self.obtain_datass
        checkWritableFilePath = self._checkWritableFilePath
        valid_exportImagePath = (
            [
                s
                for s in [exportImagePath, default_exportImagePath]
                if checkWritableFilePath(s)
            ]
            + [None]
        )[0]
        facecolor = kwargs.get("facecolor")
        subplots_in = kwargs.get("subplots_in")

        # plots_diff = set(plots)-{"eeu", "eem", "ratio", "del", "ld"}
        # if len(plots_diff) >= 1:
        #    print(", ".join(list(plots_diff))+" are not appropriate")

        fig = plt.figure()
        fig.patch.set_facecolor(facecolor)
        datass_valid = datass or obtain_datass(plots=plots)

        subplots = []

        # set height ratios for sublots
        if subplots_in is None or len(subplots_in) < len(plots):
            gss = gridspec.GridSpec(
                len(plots),
                1,
                height_ratios=[
                    2 if self._categorizePlotType(s, "big") else 1 for s in plots
                ],
            )
        else:
            gss = [None for _ in range(len(plots))]

        for ind_gs, (gs_tmp, plotType_input) in enumerate(zip(gss, plots)):
            plotType_orig = self.obtainOriginalPlotType(
                plotType_input, flag_strict=True
            )
            plotType = plotType_orig

            dataInfos = self._dictGet(datass_valid, [plotType_orig, plotType_input])
            datas = dataInfos.get("data") if isinstance(dataInfos, dict) else None
            info = dataInfos.get("info", {}) if isinstance(dataInfos, dict) else None
            if datas is None:
                continue

            # the fisrt subplot
            if ind_gs == 0:
                if gs_tmp is None:
                    ax = subplots_in[ind_gs]
                else:
                    ax = plt.subplot(gs_tmp)
                subplots.append(ax)
            else:
                if gs_tmp is None:
                    ax = subplots_in[ind_gs]
                else:
                    ax = plt.subplot(gs_tmp, sharex=subplots[0])
                subplots.append(ax)

            ax.patch.set_facecolor(facecolor)

            # set scale
            logFunc_dict = {"x": ax.set_xscale, "y": ax.set_yscale}
            for key_xy in ["x", "y"]:
                if info.get("log", {}).get(key_xy, False) is True:
                    logFunc_dict[key_xy]("log")

            if self._categorizePlotType(plotType, "log"):
                ax.set_yscale("log")
                # pass
            elif plotType_orig in ["ratio"]:
                ax.axhline(1, ls=":", lw=1.0, color="black", alpha=1, zorder=0)
            elif plotType_orig in ["delchi"]:
                ax.axhline(0, ls=":", lw=1.0, color="black", alpha=1, zorder=0)

            # set lim
            y_lim = self._dictGet(y_lims, [plotType_input, plotType_orig])
            self.set_limsFromDatas(datas, ax=ax, x_lim=x_lim, y_lim=y_lim)

            # set label
            labels = info.get("labels", "")
            plt.subplots_adjust(hspace=0.0)
            if not gs_tmp == gss[-1]:
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_xminorticklabels(), visible=False)
            else:
                xlabel = (
                    kwargs.get("xlabel")
                    if isinstance(kwargs.get("xlabel"), str)
                    else labels.get("x", "Energy (keV)")
                )
                ax.set_xlabel(xlabel)
            # ylabel_dic = {}
            ylabel_dic_default = {
                "eeufspec": r"keV$\mathdefault{^2}$ (Photons cm$^\mathdefault{-2}$ s$^\mathdefault{-1}$ keV$^\mathdefault{-1}$)",
                "eemmodel": "$EF_\mathdefault{E}$ (keV$^2$)",
                "ldata": "normalized counts s$^\mathdefault{-1}$ keV$^\mathdefault{-1}$",
                "ratio": "ratio",
                "delchi": "residual",
            }
            ylabel_dic_fromKwargs = (
                kwargs.get("ylabel_dic")
                if isinstance(kwargs.get("ylabel_dic"), dict)
                else {}
            )
            ylabel_dic = {**ylabel_dic_default, **ylabel_dic_fromKwargs}
            ylabel = self._dictGet(
                ylabel_dic, [plotType_input, plotType_orig], labels.get("y", "")
            )
            ylabel_fontsize_fromKwargs = self._dictGet(
                kwargs.get("ylabel_fontsize_dic"), [plotType_input, plotType_orig]
            )
            ylabel_fontsize = (
                ylabel_fontsize_fromKwargs
                if self._judgeReal(ylabel_fontsize_fromKwargs)
                else 10
            )
            ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)

            fig.align_labels()

            # ## plot eeu
            scatters = []
            for plotGroup, data_tmp in datas.items():
                # data
                xs = data_tmp["xs"]
                marker = (markers + ["o"] * (plotGroup + 1))[plotGroup - 1]
                color = (colors + ["black"] * (plotGroup + 1))[plotGroup - 1]
                # plotType in {"eeu", "eem", "ratio", "del", "ld"} and
                if kwargs.get("flag_dataPlot") is True and "ys" in data_tmp.keys():
                    ys = data_tmp["ys"]
                    xe = data_tmp.get("xe", [0] * len(xs))
                    ye = data_tmp.get("ye", [0] * len(ys))
                    marker_size = kwargs.get("marker_size_data")  # 0
                    elinewith = kwargs.get("elinewidth_data")  # 1
                    scatter_tmp = ax.scatter(
                        xs, ys, marker=marker, color=color, s=marker_size
                    )
                    scatters.append(scatter_tmp)
                    ax.errorbar(
                        xs,
                        ys,
                        yerr=ye,
                        xerr=xe,
                        capsize=0,
                        fmt=marker,
                        markersize=0,
                        ecolor=color,
                        markeredgecolor="none",
                        color="none",
                        elinewidth=elinewith,
                    )

                # plotType in {"eeu", "eem", "ld"} and
                if (
                    kwargs.get("flag_modelPlot") is True
                    and "ys_model" in data_tmp.keys()
                ):
                    ys_model = data_tmp.get("ys_model", [])
                    # plt.plot(xs, ys_model, color=color)
                    # plt.scatter(xs, ys_model, color=color, marker="_")
                    xe = data_tmp.get("xe", [0] * len(xs))
                    xems = sorted(zip(xs, xe, ys_model), key=lambda x: x[0])
                    xs_tmp = [s[0] for s in xems]
                    xe_tmp = [s[1] for s in xems]
                    ys_tmp = [s[2] for s in xems]

                    xs_step = [x - xe_tmp for x, xe_tmp in zip(xs_tmp, xe_tmp)] + [
                        xs_tmp[-1] + xe_tmp[-1]
                    ]
                    ys_step = [ys_tmp[0]] + ys_tmp

                    marker_size = kwargs.get("marker_size_model")  # 0
                    elinewith = kwargs.get("elinewidth_model")  # 0.5

                    if kwargs.get("flag_modelStep") is True:
                        ax.step(
                            xs_step,
                            ys_step,
                            color=color,
                            linewidth=0.5,
                            solid_joinstyle="miter",
                            markeredgecolor="none",
                        )
                    else:
                        ax.scatter(xs, ys_model, color=color, marker="_", s=marker_size)
                        ax.errorbar(
                            xs,
                            ys_model,
                            xerr=xe,
                            capsize=0,
                            fmt=marker,
                            markersize=0,
                            ecolor=color,
                            markeredgecolor="none",
                            color="none",
                            elinewidth=elinewith,
                        )

                if (
                    kwargs.get("flag_compPlot") is True
                    and "ys_comps" in data_tmp.keys()
                ):
                    ys_comps = data_tmp.get("ys_comps", [[]])
                    for ys_comp in ys_comps:
                        ax.plot(xs, ys_comp, linestyle="dotted", color=color)
                # plt.plot()
            # legend = self._dictGet(legends_dic, [plotType_input, plotType_orig], [])+[""]*(plotGroup+1)
            # scatters_now = [scatters[ind] for ind, legend_name in enumerate(
            #    legend) if len(legend_name) > 0]
            # legend_now = [
            #    legend_name for legend_name in legend if len(legend_name) > 0]
            # if len(legend_now) > 0:
            # legends_sort_tmp = legends_sort+[plotGroup+1]*(plotGroup+1)
            # scatters_tmp = sorted(
            #    scatters_now, key=lambda x: legends_sort[scatters_now.index(x)])
            # legend_tmp = sorted(
            #    legend_now, key=lambda x: legends_sort[legend_now.index(x)])
            ax.legend(fontsize=15, frameon=False, handletextpad=0, markerscale=3)

        # set title
        title = kwargs.get("title")
        if isinstance(title, str):
            subplots[0].set_title(title)

        if valid_exportImagePath is not None:
            fig.savefig(
                valid_exportImagePath, dpi=300, bbox_inches="tight", pad_inches=0.05
            )
            print(f"Figure is saved as {valid_exportImagePath}")
        return fig, subplots

    def plot_datass_fromXcms(
        self,
        xcms=[],
        title_func=None,
        flag_titleIsXcmpath=True,
        flag_addTitleReducedChisq=True,
        exportImagePath_func=None,
        **kwargs_in,
    ):
        return_dic = {}
        for xcm_path in xcms:
            loadSuccess = self.loadXcm(xcm_path=xcm_path)
            if loadSuccess is False:
                print(f"Failed to load {xcm_path}")
                continue

            kwargs_add = {}
            if callable(title_func):
                kwargs_add["title"] = title_func(xcm_path)
            elif flag_titleIsXcmpath is True:
                kwargs_add["title"] = os.path.basename(xcm_path)
            if flag_addTitleReducedChisq is True:
                stat = self.Fit.statistic
                dof = self.Fit.dof
                kwargs_add["title"] += "\n" + f"{stat:.3g} / {dof} = {stat/dof:.3g}"

            if callable(exportImagePath_func):
                kwargs_add["exportImagePath"] = exportImagePath_func(xcm_path)

            kwargs_now = {**kwargs_in, **kwargs_add}
            fig, subplots = self.plot_datass(**kwargs_now)
            return_dic[xcm_path] = {"fig": fig, "subplots": subplots}
        return return_dic

    # # parameter

    def obtainParamsPerComp(self, model):
        comps = [model.__dict__[s] for s in model.componentNames]
        return sum(
            [
                [{"param": comp.__dict__[s], "comp": comp} for s in comp.parameterNames]
                for comp in comps
            ],
            [],
        )

    def _obtainELF_list(self, error, frozen, link, order_value=5):
        error_str = [format(s, f".{order_value}g") for s in error]
        elf = (
            ["frozen", "null"]
            if frozen
            else ([link, "null"] if link != "" else error_str)
        )
        return elf

    # ------------- from texttable : begin------------
    def _to_float(self, x):
        if x is None:
            return None
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    def _judge_format(self, x):
        f = self._to_float(x)
        judged_dict = {
            "exp": abs(f) > 1e8,
            "text": f != f,  # NaN
            "int": f - round(f) == 0,
            "float": True,
        }
        return [k for k, v in judged_dict.items() if v is True][0]

    def _format_auto(self, x, order_value=5):
        format_judged = self._judge_format(x)
        if format_judged == "exp":
            return format(x, f".{order_value}e")
        elif format_judged == "text":
            return str(x)
        elif format_judged == "int":
            return str(int(x))
        else:  # float
            return format(x, f".{order_value}f")

    # ------------- from texttable : end ------------

    def obtainInfoParam(
        self, param, info_comp={}, info_model={}, flag_forPrint=False, order_value=5
    ):
        ind_param_start = info_model.get("startParIndex", None)
        _ind_dict = {
            "ind_param": param.index,
            "ind_param_total": param.index + ind_param_start - 1
            if ind_param_start is not None
            else "None",
            "ind_comp": info_comp.get("ind", "None"),
            "ind_model": info_model.get("ind", "None"),
        }
        _info_param_dict = {
            "ind": {
                "i_p": _ind_dict["ind_param"],
                "i_pT": _ind_dict["ind_param_total"],
                "i_c": _ind_dict["ind_comp"],
                "i_m": _ind_dict["ind_model"],
            }
            if flag_forPrint is True
            else _ind_dict,
            "main": {
                "name_comp": info_comp.get("name", "None"),
                "name_param": param.name,
                "unit": param.unit,
                "value": format(param.values[0], f".{order_value}g")
                if flag_forPrint is True
                else param.values[0],
            },
            "forNotPrint": {}
            if flag_forPrint is True
            else {"error": param.error[:2], "frozen": param.frozen, "link": param.link},
            "elf": {
                f"elf_{ind_v}": v
                for ind_v, v in enumerate(
                    self._obtainELF_list(
                        error=param.error[:2],
                        frozen=param.frozen,
                        link=param.link,
                        order_value=order_value,
                    )
                )
            },
        }
        return {
            **_info_param_dict["ind"],
            **_info_param_dict["main"],
            **_info_param_dict["forNotPrint"],
            **_info_param_dict["elf"],
        }

    def obtainInfoParamsAll(
        self, flag_forPrint=False, flag_oneOnly=False, order_value=5, flag_nest=True
    ):
        AllModels = self.AllModels
        AllData = self.AllData

        nGroups = AllData.nGroups
        models = [AllModels(ind_model) for ind_model in range(1, nGroups + 1)]

        _info_paramss = [
            [
                self.obtainInfoParam(
                    paramAndComp["param"],
                    info_comp={
                        "ind": model.componentNames.index(paramAndComp["comp"].name)
                        + 1,
                        "name": paramAndComp["comp"].name,
                    },
                    info_model={
                        "startParIndex": model.startParIndex,
                        "ind": ind_model + 1,
                    },
                    flag_forPrint=flag_forPrint,
                    order_value=order_value,
                )
                for paramAndComp in self.obtainParamsPerComp(model)
            ]
            for ind_model, model in enumerate(models)
        ]
        info_paramss = [_info_paramss[0]] if flag_oneOnly is True else _info_paramss
        if flag_nest is True:
            return info_paramss
        else:
            return sum(info_paramss, [])

    def obtainInfoParamsAll_df(self, **kwargs):
        info_paramss_tmp = self.obtainInfoParamsAll(flag_nest=True, **kwargs)
        df_tmp = pd.DataFrame(sum(info_paramss_tmp, []))
        df_tmp["xcm"] = os.path.basename(self.xcm)
        return df_tmp

    def showParamsAll(self, flag_oneOnly=False, max_width=100, order_value=5):
        Fit = self.Fit
        stat = Fit.statistic
        dof = Fit.dof
        chisq = stat / dof

        info_paramss = self.obtainInfoParamsAll(
            flag_forPrint=True, flag_oneOnly=flag_oneOnly, order_value=order_value
        )

        # table
        table = texttable.Texttable(max_width=max_width)
        table.set_deco(texttable.Texttable.HEADER)
        table.set_cols_align(["r", "r", "r", "r", "l", "l", "l", "l", "l", "l"])
        table.set_cols_dtype(["i", "i", "i", "i", "t", "t", "t", "t", "t", "t"])
        header = list(info_paramss[0][0].keys())
        table.add_rows(
            [header]
            + sum(
                sum(
                    [
                        [
                            [[""] * (len(header))],
                            [list(s.values()) for s in info_params],
                        ]
                        for ind_model, info_params in enumerate(info_paramss)
                    ],
                    [],
                ),
                [],
            )
        )
        print(
            table.draw()
            + "\n\t\tReduced Chi-Squared: {:.5g} / {} = {:.5g}".format(stat, dof, chisq)
        )

    def showParamsAll_xspec(self, flag_oneOnly=False):
        AllModels = self.AllModels
        Fit = self.Fit

        if flag_oneOnly is True:
            AllModels(1).show()
        else:
            AllModels.show()
        Fit.show()

    # # calc flux

    def calcFluxLumin(self, cmdStr, flag_flux=True, flag_omit=True):
        AllModels = self.AllModels
        AllData = self.AllData
        nGroups = AllData.nGroups

        if flag_flux is True:
            AllModels.calcFlux(cmdStr)
        else:
            AllModels.calcLumin(cmdStr)
        output = [
            AllModels(ind_model).flux
            if flag_flux is True
            else AllModels(ind_model).lumin
            for ind_model in range(1, nGroups + 1)
        ]

        if flag_omit is True and all(s == output[0] for s in output):
            return [output[0]]

        return output

    def calcFluxLumin_unabs(self, x_lim, compParams_abs=None, **kwargs):
        if compParams_abs is not None:
            compParams_old = self.set_params(compParams_abs)
        else:
            compParams_old = None
        ene_range = " ".join(map(str, x_lim))  # "1e-3 20"
        self.AllData.dummyrsp(ene_range)
        if "cmdStr" not in kwargs.keys():
            kwargs["cmdStr"] = ene_range
        output = self.calcFluxLumin(**kwargs)
        print(output)
        self.set_params(compParams_old)

    def performErrorAll(self, flag_oneOnly=True):
        self.Fit.perform()

        param_infos_tmp = self.obtainInfoParamsAll(flag_oneOnly=flag_oneOnly)
        param_infos = sum(param_infos_tmp, [])
        df_tmp = pd.DataFrame(param_infos)
        inds_param_free = df_tmp[df_tmp["elf_1"] != "null"]["ind_param_total"].to_list()
        self.Fit.error(" ".join(map(str, inds_param_free)))

    def obtainInfoParamsForExport(
        self,
        header_selected=[],
        flag_showHeader=True,
        flag_oneOnly=False,
        flag_print=True,
        flag_copy=True,
    ):
        Fit = self.Fit
        info_stat = {
            "stat": Fit.statistic,
            "dof": Fit.dof,
            "chisq": Fit.statistic / Fit.dof,
        }

        info_paramss = self.obtainInfoParamsAll(
            flag_forPrint=True, flag_oneOnly=flag_oneOnly
        )

        add_info = {
            "xcm": os.path.basename(self.xcm),
            "chisq": "{}".format(info_stat["chisq"]),
            "stat/dof": "{}/{}".format(info_stat["stat"], info_stat["dof"]),
        }

        info_paramss[0][0] = {**info_paramss[0][0], **add_info}
        if header_selected is not None and hasattr(header_selected, "__iter__"):
            info_paramss = [
                [{k: v for k, v in info_param.items()} for info_param in info_params]
                for info_params in info_paramss
            ]
        table_header = (
            "\t".join(info_paramss[0][0].keys()) + "\n"
            if flag_showHeader is True
            else ""
        )
        table_body = "\n".join(
            [
                "\n".join(
                    [
                        "\t".join(map(str, list(info_param.values())))
                        for info_param in info_params
                    ]
                )
                for info_params in info_paramss
            ]
        )

        table_total = table_header + table_body

        if flag_copy is True:
            pyperclip.copy(table_total)
        if flag_print is True:
            string_method = "copied to the clipboard and " if flag_copy is True else ""
            print(
                f"\t\tThe following table is{string_method} returned, you will obtain the table by pasting it on spreadsheet.\n\n"
            )
            print(table_total)
        return table_total

    # # plot unabs
    def set_params(self, compParams):
        model_tmp = self.AllModels(1)
        compParams_old = []
        for compParam in compParams:
            if len(compParam) < 2:
                continue
            compName = compParam[0]
            paramName = compParam[1]
            new_value = 0 if len(compParam) == 2 else compParam[2]
            if not compName in model_tmp.componentNames:
                continue
            comp_tmp = model_tmp.__dict__[compName]
            if not paramName in comp_tmp.parameterNames:
                continue
            param_tmp = comp_tmp.__dict__[paramName]
            compParams_old.append([compName, paramName, param_tmp.values])
            param_tmp.values = new_value
        return compParams_old

    def ignore_specified_range(self, ignore_x_lim, xs):  # , *args
        if not hasattr(ignore_x_lim, "__iter__"):
            return xs  # , *args
        judged_xs = []
        # arr_judge=[]
        for val_x in xs:
            judge = True
            for x_lim in ignore_x_lim:
                if val_x >= min(x_lim) and val_x <= max(x_lim):
                    judge = False
                    break
            # arr_judge.append(judge)
            if judge is True:
                judged_xs.append(val_x)
            else:
                judged_xs.append(None)

        # arr_ret=[]
        # for arg in args:
        #    if len(arg)!=len(arr_judge):
        #        arr_ret.append(arg)
        #        continue
        #    arr_tmp=[]
        #    for judge, val_a in zip(arr_judge,arg):
        #        if judge is True:
        #            arr_tmp.append(val_a)
        #        else:
        #            arr_tmp.append(None)
        #    arr_ret.append(arr_tmp)
        # return judged_xs, *arr_ret
        return judged_xs

    # ## obtain datasss for Unabs plot

    def obtain_datasss_s_forUnabsPlot_fromXcmPaths(self, xcm_paths, **kwargs):
        return {
            xcm_path: self.obtain_datasss_forUnabsPlot(xcm_path, **kwargs)
            for xcm_path in xcm_paths
        }

    def obtain_datasss_forUnabsPlot(
        self, xcm_path, x_lim=None, y_lim=None, compParams_abs=[]
    ):
        plot_type = "eeufspec"
        load_status = self.loadXcm(xcm_path)
        if load_status is not True:
            return False
        self.Fit.perform()
        stat = self.Fit.statistic
        dof = self.Fit.dof
        datass_abs = self.obtain_datass(["eeu", "eem"])

        # set lim
        lims = self.set_limsFromDatas(
            datass_abs[plot_type]["data"], ax=None, x_lim=x_lim, y_lim=y_lim
        )
        lims_unabs = self.set_limsFromDatas(
            datass_abs[plot_type]["data"],
            ax=None,
            x_lim=x_lim,
            y_lim=y_lim,
            flag_inclusive=True,
        )
        x_lim = lims["x"]
        x_lim_large = lims_unabs["x"]

        # unabs

        x_lim_large_keV = sorted(
            [
                (s * astropy.units.__dict__[self.Plot.xAxis])
                .to(astropy.units.keV, equivalencies=astropy.units.spectral())
                .value
                for s in x_lim_large
            ]
        )
        self.set_params(compParams_abs)
        ene_range = " ".join(map(str, x_lim_large_keV))  # "1e-3 20"
        self.AllData.dummyrsp(ene_range)

        datass_unabs = self.obtain_datass(["eem"])

        return {
            "info": {
                "x_lim": x_lim,
                "y_lim": y_lim,
                "xcm_path": xcm_path,
                "stat": stat,
                "dof": dof,
                "fitsNames": self.obtain_fitsNames(),
            },
            "abs": datass_abs,
            "unabs": datass_unabs,
        }

    def plot_unabsModelAndDatass(
        self,
        xcm_path=None,
        compParams_abs=[],
        x_lim=None,
        y_lim=None,
        exportImagePath=None,
        **kwargs_in,
    ):
        plt = self.plt

        kwargs_default = {
            "linecolors": {},
            "linestyles": {},
            "colors": [],
            "title": None,
            "xlabel": None,
            "ylabel": None,
            "facecolor": "white",
            "markers": [],
            "subpltos_in": None,
            "zorders": {},
            "ignore_x_lims": {},
            "flag_dataIsFilled": True,
            "datasss": None,
        }
        types_forNone_tmp = {
            str: ["title", "xlabel", "ylabel"],
            list: ["subplots_in"],
            dict: ["datass_for_unabs"],
        }
        kwargs_types_forNone = {
            k: instance for instance, keys in types_forNone_tmp.items() for k in keys
        }
        kwargs = {**kwargs_default, **kwargs_in}

        facecolor = kwargs.get("facecolor")
        subplots_in = kwargs.get("subplots_in")

        plot_type = "eeufspec"
        default_infos = {
            "data_elinewith": 1,
            "data_alpha": 0.8,
            "data_marker_alpha": 0.6,
            "data_errorbar_alpha": 0.8,
            "linecolor": "dimgray",
            "linestyle": "-",
            "color": "royalblue",
            "linestyles": ["--", ":", "-.", (0, (1, 0))],
            "zorders": {"model": -100, "comps": 100, "data": 200},
            "flag_dataIsFilled": True,
        }

        linecolors = kwargs.get("linecolors", {})
        linestyles = kwargs.get("linestyles", {})
        colors = kwargs.get("colors", [])
        markers = kwargs.get("markers", [])
        zorders = kwargs.get("zorders", {})
        ignore_x_lims = kwargs.get("ignore_x_lims") or {}
        flag_dataIsFilled = bool(kwargs.get("flag_dataIsFilled"))
        datasss_in = kwargs.get("datasss", {})
        # info_in=datasss_in.get("info")

        # check kwargs
        if False:
            keys_invalidInstance = []
            for key, val in kwargs.items():
                flag_isNone = val is None
                flag_isValidInstance = isinstance(
                    val,
                    kwargs_types_forNone.get(key, kwargs_default.get(key).__class__),
                )
                if not (flag_isNone or flag_isValidInstance):
                    keys_invalidInstance.append(key)
            if len(keys_invalidInstance) > 0:
                message = "Invalid Instances: " + "\n".join(
                    [
                        f"\t{k}\t:\t{kwargs_types_forNone.get(key, kwargs_default.get(key).__class__).__name__}"
                        for k in keys_invalidInstance
                    ]
                )
                print(message)
                return False
        # fig, ax
        if subplots_in is None:
            fig, ax = plt.subplots()
        else:
            fig = None
            ax = subplots_in[0]
        ax.patch.set_facecolor(facecolor)
        ax.set_xscale("log")
        ax.set_yscale("log")

        flag_datasssInIsValid = all(s in datasss_in.keys() for s in ["abs", "unabs"])

        if flag_datasssInIsValid is False:
            datasss_in = self.obtain_datasss_forUnabsPlot(
                xcm_path, x_lim, y_lim, compParams_abs
            )
        datass_abs = datasss_in.get("abs")
        datass_unabs = datasss_in.get("unabs")
        data_unabs = datass_unabs["eemodel"]["data"][1]
        info_in = datasss_in.get("info")
        if info_in is not None:
            if x_lim is None:
                x_lim = info_in.get("x_lim")
            if y_lim is None:
                y_lim = info_in.get("y_lim")
        # set lim
        lims = self.set_limsFromDatas(
            datass_abs[plot_type]["data"], ax=ax, x_lim=x_lim, y_lim=y_lim
        )
        # lims_unabs = self.set_limsFromDatas(datass_abs[plot_type]["data"], ax=None, x_lim=x_lim, y_lim=y_lim, flag_inclusive=True)
        x_lim = lims["x"]
        # x_lim_large=lims_unabs["x"]
        unabs_xs = data_unabs["xs"]
        unabs_ys_model = data_unabs["ys_model"]
        unabs_eem_func = scipy.interpolate.interp1d(unabs_xs, unabs_ys_model)

        if "ys_model" in data_unabs.keys():
            linecolor = linecolors.get(
                "model", default_infos.get("linecolor", "dimgray")
            )
            linestyle = linestyles.get("model", default_infos.get("linestyle", "-"))
            zorder = zorders.get("model", default_infos.get("zorders").get("model"))
            ig_xlim = ignore_x_lims.get("model", None)
            xs1 = self.ignore_specified_range(ig_xlim, unabs_xs)
            ys1 = unabs_ys_model
            # ax.plot(unabs_xs, unabs_ys_model, color=linecolor, linestyle=linestyle, zorder=zorder)
            ax.plot(xs1, ys1, color=linecolor, linestyle=linestyle, zorder=zorder)

        if "ys_comps" in data_unabs.keys() and "compNames" in data_unabs.keys():
            zorder = zorders.get("comp", default_infos.get("zorders").get("comp"))
            for ind_comp, (compName, ys) in enumerate(
                zip(data_unabs["compNames"], data_unabs["ys_comps"])
            ):
                linestyles_default = default_infos.get("linestyles")
                linecolor = linecolors.get(compName, default_infos.get("linecolor"))
                linestyle = linestyles.get(
                    compName, linestyles_default[ind_comp % len(linestyles_default)]
                )
                ig_xlim = ignore_x_lims.get("comp", None)
                xs1 = self.ignore_specified_range(ig_xlim, unabs_xs)
                ys1 = ys
                # ax.plot(unabs_xs, ys, color=linecolor, linestyle=linestyle, label=compName, zorder=zorder)
                ax.plot(
                    xs1,
                    ys1,
                    color=linecolor,
                    linestyle=linestyle,
                    label=compName,
                    zorder=zorder,
                )
            ax.legend(edgecolor="none")

        colors_dict = dict(enumerate(colors))
        markers_dict = dict(enumerate(markers))
        zorder = zorders.get("data", default_infos.get("zorders").get("data"))
        for num_data, data in datass_abs[plot_type]["data"].items():
            xs = data["xs"]
            xe = data["xe"]
            ys = data["ys"]
            ye = data["ye"]
            ys_model = data["ys_model"]
            unabs_ys = [y / m * unabs_eem_func(x) for x, y, m in zip(xs, ys, ys_model)]
            unabs_ye = [
                e / m * unabs_eem_func(x) for x, y, e, m in zip(xs, ys, ye, ys_model)
            ]
            color = colors_dict.get(
                int(num_data) - 1, default_infos.get("color", "royalblue")
            )
            marker = markers_dict.get(
                int(num_data) - 1, default_infos.get("marker", None)
            )
            elinewidth = kwargs.get(
                "data_elinewidth", default_infos.get("data_elinewidth")
            )
            marker_alpha = kwargs.get(
                "data_marker_alpha",
                kwargs.get("data_alpha", default_infos.get("data_marker_alpha")),
            )
            errorbar_alpha = kwargs.get(
                "data_errorbar_alpha",
                kwargs.get("data_alpha", default_infos.get("data_errorbar_alpha")),
            )
            size = kwargs.get("data_size", default_infos.get("data_size", 20))

            ig_xlim = ignore_x_lims.get(num_data, ignore_x_lims.get("data"))
            xs1 = self.ignore_specified_range(ig_xlim, xs)
            xe1, ys1, uy1, uye1 = xe, ys, unabs_ys, unabs_ye

            if marker is not None:
                # ax.scatter(xs1, unabs_ys, marker=marker, alpha=alpha, color="none", edgecolor=color, s=size, zorder=zorder)
                if flag_dataIsFilled is True:
                    color1 = color
                else:
                    color1 = "none"
                ax.scatter(
                    xs1,
                    uy1,
                    marker=marker,
                    alpha=marker_alpha,
                    color=color1,
                    edgecolor=color,
                    s=size,
                    zorder=zorder,
                )
            # ax.errorbar(xs, unabs_ys, xerr=xe, yerr=unabs_ye, markeredgecolor="none", color="none",
            #            ecolor=color, elinewidth=elinewidth, alpha=alpha, zorder=zorder)
            ax.errorbar(
                xs1,
                uy1,
                xerr=xe1,
                yerr=uye1,
                markeredgecolor="none",
                color="none",
                ecolor=color,
                elinewidth=elinewidth,
                alpha=errorbar_alpha,
                zorder=zorder,
            )

            # plt.errorbar(xs, unabs_ys, unabs_ye)

        title_cand = os.path.basename(xcm_path).replace(".xcm", "")
        title = kwargs.get("title", title_cand)
        ax.set_title(title)
        ax.set_xlabel(datass_abs[plot_type]["info"]["labels"]["x"])
        ax.set_ylabel(datass_abs[plot_type]["info"]["labels"]["y"])

        valid_exportImagePath = (
            [
                s
                for s in [exportImagePath, self.default_exportImagePath]
                if self._checkWritableFilePath(s)
            ]
            + [None]
        )[0]

        if valid_exportImagePath is not None and fig is not None:
            fig.savefig(
                valid_exportImagePath, dpi=300, bbox_inches="tight", pad_inches=0.05
            )
            print(f"Figure is saved as {valid_exportImagePath}")

        return fig, ax
