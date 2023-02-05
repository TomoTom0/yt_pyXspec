# import
import os
import re
import sys
#import shutil
import glob2
import texttable
import warnings
import xspec
import numpy as np
import matplotlib.pyplot as _plt
from matplotlib import gridspec


class ytpx():
    # # init
    def __init__(self, dir_path=None):
        self.dir_path = None
        self.chDir(dir_path=dir_path)
        self.findXcm(dir_path=dir_path)

        self.Plot = xspec.Plot
        self.AllModels = xspec.AllModels
        self.AllData = xspec.AllData
        self.Xset = xspec.Xset
        self.Fit = xspec.Fit
        self.plt = _plt
        self.xspec = xspec

        self.__stdout__ = sys.stdout
        self.default_exportImagePath = None
        self.xcm = None
        self.__initMatplotlibRcParams()
        self.__initXspecPlot()
        # self.gridspec=_gridspec

    def chDir(self, dir_path=None):
        if os.path.isdir(dir_path):
            print(f"change directory to {dir_path}")
            os.chdir(dir_path)
            self.dir_path = dir_path

    def findXcm(self, keyword=None, dir_path=None):
        valid_dir_path = dir_path or self.dir_path
        valid_keyword = keyword or "*.xcm"
        xcms = sorted(glob2.glob(valid_dir_path+"/"+valid_keyword)
                      if valid_dir_path is not None else [])
        self.xcms = xcms
        return xcms

    # Disable
    def blockPrint(self):
        sys.stdout = open(os.devnull, "w")

    # Restore
    def enablePrint(self):
        sys.stdout = self.__stdout__

    def __initXspecPlot(self):
        Plot = self.Plot
        Plot.xAxis = "keV"
        Plot.addCommand("time off")
        Plot.addCommand("font roman")

        Plot.addCommand("cs 2")
        Plot.addCommand("lw 5")
        Plot.addCommand("win 1")
        Plot.addCommand("loc 0.05 0.05 1 0.95")
        Plot.addCommand("win 2")
        Plot.addCommand("loc 0.05 0.05 1 0.95")

    def __initMatplotlibRcParams(self):
        plt = self.plt
        plt.rcParams["figure.figsize"] = (3.14*2, 3.14*2*0.7)
        plt.rcParams["figure.dpi"] = 100  # 画像保存するときは300に

        plt.rcParams["font.family"] = "Times New Roman"  # 全体のフォントを設定
        plt.rcParams["xtick.direction"] = "in"  # x軸の目盛線を内向きへ
        plt.rcParams["ytick.direction"] = "in"  # y軸の目盛線を内向きへ
        plt.rcParams["xtick.minor.visible"] = True  # x軸補助目盛りの追加
        plt.rcParams["ytick.minor.visible"] = True  # y軸補助目盛りの追加
        plt.rcParams["xtick.major.width"] = 1.5  # x軸主目盛り線の線幅
        plt.rcParams["ytick.major.width"] = 1.5  # y軸主目盛り線の線幅
        plt.rcParams["xtick.minor.width"] = 1.0  # x軸補助目盛り線の線幅
        plt.rcParams["ytick.minor.width"] = 1.0  # y軸補助目盛り線の線幅
        plt.rcParams["xtick.major.size"] = 10  # x軸主目盛り線の長さ
        plt.rcParams["ytick.major.size"] = 10  # y軸主目盛り線の長さ
        plt.rcParams["xtick.minor.size"] = 5  # x軸補助目盛り線の長さ
        plt.rcParams["ytick.minor.size"] = 5  # y軸補助目盛り線の長さ
        plt.rcParams["font.size"] = 14  # フォントの大きさ
        plt.rcParams["axes.linewidth"] = 1.5  # 囲みの太さ

    def loadXcm(self, xcm_path):
        if not os.path.isfile(xcm_path):
            return False
        self.blockPrint()
        self.Xset.restore(xcm_path)
        self.enablePrint()
        self.xcm = xcm_path
        return True

    # # obtain_datass
    def obtain_datass_s_fromXcms(self, xcms=[], **kwargs_in):
        # plots
        datas_dic = {}
        for xcm_path in xcms:
            self.loadXcm(xcm_path)
            datas_now = self.obtain_datas(**kwargs_in)
            datas_dic[xcm_path] = datas_now
        return datas_dic

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

        for plotWindow_tmp, plot_type in enumerate(plots):
            plotWindow = plotWindow_tmp+1
            datas = {}
            datas_info = {
                "labels":{
                        key_xy: re.sub(r"\$([^\$]*)\$", r"$\\mathdefault{\1}$", label) 
                            for key_xy, label in zip(["x", "y"], Plot.labels(plotWindow))
                    },
                "log":{
                        key_xy:_log for key_xy, _log in zip(["x", "y"], [Plot.xLog, Plot.yLog])
                    },
                "model":AllModels(1).expression,
                "plotWindow":plotWindow
            }

            #xs, ys, xe, ye, ys_model, ys_comps = [[]]*6
            for plotGroup in range(1, AllData.nGroups+1):
                dataFuncs_dict={
                    "xs":Plot.x,
                    "ys":Plot.y,
                    "xe":Plot.xErr,
                    "ye":Plot.yErr,
                    "ys_model":Plot.model
                }
                datas_data={}
                for key_data, dataFunc in dataFuncs_dict.items():
                    try:
                        warnings.simplefilter("ignore")
                        data_obtained=dataFunc(plotGroup, plotWindow)
                        warnings.resetwarnings()
                        datas_data[key_data]=data_obtained
                    except Exception as e:
                        pass

                model = AllModels(plotGroup)

                # obtain comps in models
                comps_obtained = []
                compNames = []
                addComps = self._obtainAdditiveComps(model)
                if len(addComps) <= 1:
                    datas_comp = {}
                else:
                    for ind_compAdd, compAdd in enumerate(addComps):
                        try:
                            comp_tmp = Plot.addComp(
                                ind_compAdd+1, plotGroup, plotWindow)
                            # execlude components with only 0
                            if all(s==0 for s in comp_tmp):
                                continue
                            comps_obtained.append(comp_tmp)
                            compNames.append(compAdd.name)
                        except:
                            break
                    datas_comp = {"ys_comps": comps_obtained, "compNames": compNames}
                datas[plotGroup] = {**datas_data, **datas_comp}
            datass[plot_type] = {"data":datas, "info":datas_info}

        return datass

    def obtain_datass_dic(self, xcm_paths_dic, plots_dic):
        Xset = self.Xset
        datass_dic = {}
        for key_xcms, xcm_paths in xcm_paths_dic.items():
            datass_s = {}
            for key_xcm, xcm_path in xcm_paths.items():
                self.loadXcm(xcm_path)
                datass_s[key_xcm] = self.obtain_datas(plots=plots_dic[key_xcms])

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
    def __checkWritableFilePath(self, path):
        if path is None:
            return False
        isFile_OK = (os.path.isfile(path) and os.access(path, os.W_OK))
        isNotExist_OK = (os.path.isdir(os.path.split(path)
                         [0]) and not os.path.exists(path))
        return isFile_OK or isNotExist_OK

    def __extractValues(self, datas):
        sum_dic = {}
        for key_xy in ["x", "y"]:
            valss = [s.get(key_xy+"s", []) for s in datas.values()]
            errss = [s.get(key_xy+"e", []) for s in datas.values()]
            models = [s.get(key_xy+"s_model" , []) for s in datas.values()]
            sum_tmp = []
            for vals, errs in zip(valss, errss):
                if len(errs) == len(vals):
                    sum_tmp += list(np.array(vals*2)+np.array(errs*2)
                                    * np.array([1]*len(errs)+[-1]*len(errs)))
                else:
                    sum_tmp += vals
            sum_dic[key_xy+"s"] = sum_tmp+sum(models, [])
            # xs_sum=sum([np.array(s["xs"]*2)+np.array(s["xe"]*2)*np.array([1]*len(s["xe"])+[-1]*len(s["xe"])) for s in data["eeu"].values()], [])
            # ys_sum=sum([np.array(s["ys"]*2)+np.array(s["ye"]*2)*np.array([1]*len(s["ye"])+[-1]*len(s["ye"])) for s in data["eeu"].values()], [])
        return sum_dic  # {"xs":xs_sum, "ys":ys_sum}

    def __obtainLim(self, values, logIsValid=True, margin_ratio=0.05):
        valid_values=[s for s in values if (not logIsValid) or s>0]
        if len(valid_values)==0:
            return []
        v_min = min(valid_values)
        v_max = max(valid_values)
        if logIsValid is True:
            margin = (np.log10(v_max)-np.log10(v_min))*margin_ratio
            return (10**(np.log10(np.array([v_min, v_max]))+np.array([-1, +1])*margin)).tolist()
        else:
            margin = (v_max-v_min)*0.05
            return ((np.array([v_min, v_max]))+np.array([-1, +1])*margin).tolist()

    # # plot datass
    def plot_datass(self,
                  plots=["eeu"],
                  x_lim=[],
                  y_lims={},
                  colors=["royalblue", "red", "olivedrab", "turquoise", "orange",
                          "chartreuse", "navy", "firebrick", "darkgreen", "darkmagenta"],
                  markers=[],
                  legends_dic={},
                  legends_sort=[],
                  datass=None,
                  exportImagePath=None, **kwargs_in):

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
            "title": None
        }
        kwargs = {**kwargs_default, **kwargs_in}

        plt = self.plt
        default_exportImagePath = self.default_exportImagePath
        obtain_datass = self.obtain_datass
        checkWritableFilePath = self.__checkWritableFilePath
        valid_exportImagePath = ([s for s in [
                                 exportImagePath, default_exportImagePath] if checkWritableFilePath(s)]+[None])[0]

        #plots_diff = set(plots)-{"eeu", "eem", "ratio", "del", "ld"}
        #if len(plots_diff) >= 1:
        #    print(", ".join(list(plots_diff))+" are not appropriate")

        fig = plt.figure()
        fig.patch.set_facecolor("white")
        datass_valid = datass or obtain_datass(plots=plots)

        subplots = []

        # set height ratios for sublots
        gs = gridspec.GridSpec(len(plots), 1, height_ratios=[
            2 if s in ["eeu", "eem", "ld"] else 1 for s in plots])

        for gs_tmp, plot_type in zip(gs, plots):
            dataInfos = datass_valid.get(plot_type, None)
            datas = dataInfos.get("data") if isinstance(dataInfos, dict) else None
            info = dataInfos.get("info", {}) if isinstance(dataInfos, dict) else None
            if datas is None:
                continue

            # the fisrt subplot
            ax = plt.subplot(gs_tmp)
            subplots.append(ax)

            # set scale
            logFunc_dict={"x":ax.set_xscale,"y":ax.set_yscale}
            for key_xy in ["x", "y"]:
                if info.get("log", {}).get(key_xy, False) is True:
                    logFunc_dict[key_xy]("log")

            if not plot_type in ["ratio", "del"]:
                #ax.set_yscale("log")
                pass
            elif plot_type in ["ratio"]:
                plt.axhline(1, ls=":", lw=1., color="black", alpha=1, zorder=0)
            elif plot_type in ["del"]:
                plt.axhline(0, ls=":", lw=1., color="black", alpha=1, zorder=0)

            # set lim
            data_sum = self.__extractValues(datas)
            scaleIsLog = {
                "x": ax.get_xscale() == "log",
                "y": ax.get_yscale() == "log"
            }
            lims_fromValue = {
                key_xy+"s": self.__obtainLim(data_sum[key_xy+"s"], logIsValid=scaleIsLog.get(key_xy, True))
                for key_xy in ["x", "y"]}
            ax.set_xlim(lims_fromValue["xs"])
            ax.set_ylim(lims_fromValue["ys"])

            y_lim = y_lims.get(plot_type, None)
            if hasattr(x_lim, "__iter__") and not len(x_lim) < 2:
                ax.set_xlim(x_lim)
            if hasattr(y_lim, "__iter__") and not len(y_lim) < 2:
                ax.set_ylim(y_lim)

            # set label
            labels=info.get("labels", "")
            plt.subplots_adjust(hspace=.0)
            if not gs_tmp == gs[-1]:
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_xminorticklabels(), visible=False)
            else:
                ax.set_xlabel(labels.get("x", "Energy (keV)"))
            ylabel_dic={}
            #ylabel_dic = {"eeu": r"keV$\mathdefault{^2}$ (Photons cm$^\mathdefault{-2}$ s$^\mathdefault{-1}$ keV$^\mathdefault{-1}$)",
            #            "eem": "$EF_\mathdefault{E}$ (keV$^2$)",
            #            "ld": "normalized counts s$^\mathdefault{-1}$ keV$^\mathdefault{-1}$",
            #            "ratio": "ratio", "del": "residual"}
            ax.set_ylabel(ylabel_dic.get(plot_type, labels.get("y", "")), fontsize=10)

            fig.align_labels()

            # ## plot eeu
            scatters = []
            for plotGroup, data_tmp in datas.items():
                #data
                xs = data_tmp["xs"]
                marker = (markers+["o"]*(plotGroup+1))[plotGroup-1]
                color = (colors+["black"]*(plotGroup+1))[plotGroup-1]
                if kwargs.get("flag_dataPlot") is True and "ys" in data_tmp.keys(): # plot_type in {"eeu", "eem", "ratio", "del", "ld"} and 
                    ys = data_tmp["ys"]
                    xe = data_tmp.get("xe", [0]*len(xs))
                    ye = data_tmp.get("ye", [0]*len(ys))
                    marker_size = kwargs.get("marker_size_data", 0)
                    elinewith = kwargs.get("elinewidth_data", 1)
                    scatter_tmp = plt.scatter(
                        xs, ys, marker=marker, color=color, s=marker_size)
                    scatters.append(scatter_tmp)
                    plt.errorbar(xs, ys, yerr=ye, xerr=xe, capsize=0, fmt=marker, markersize=0,
                                 ecolor=color, markeredgecolor="none", color="none", elinewidth=elinewith)

                if kwargs.get("flag_modelPlot") is True and "ys_model" in data_tmp.keys(): # plot_type in {"eeu", "eem", "ld"} and 
                    ys_model = data_tmp.get("ys_model", [])
                    # plt.plot(xs, ys_model, color=color)
                    # plt.scatter(xs, ys_model, color=color, marker="_")
                    xems = sorted(zip(xs, xe, ys_model), key=lambda x: x[0])
                    xs_tmp = [s[0] for s in xems]
                    xe_tmp = [s[1] for s in xems]
                    ys_tmp = [s[2] for s in xems]

                    xs_step = [
                        x-xe_tmp for x, xe_tmp in zip(xs_tmp, xe_tmp)]+[xs_tmp[-1]+xe_tmp[-1]]
                    ys_step = [ys_tmp[0]]+ys_tmp

                    marker_size = kwargs.get("marker_size_model", 0)
                    elinewith = kwargs.get("elinewidth_model", 0.5)

                    if kwargs.get("flag_modelStep") is True:
                        plt.step(xs_step, ys_step, color=color, linewidth=0.5,
                                 solid_joinstyle="miter", markeredgecolor="none")
                    else:
                        plt.scatter(xs, ys_model, color=color,
                                    marker="_", s=marker_size)
                        plt.errorbar(xs, ys_model, xerr=xe, capsize=0, fmt=marker, markersize=0,
                                     ecolor=color, markeredgecolor="none", color="none", elinewidth=elinewith)

                if kwargs.get("flag_compPlot") is True and "ys_comps" in data_tmp.keys():
                    ys_comps = data_tmp.get("ys_comps", [[]])
                    for ys_comp in ys_comps:
                        plt.plot(xs, ys_comp, linestyle="dotted",
                                 color=color)
                # plt.plot()
            legend = legends_dic.get(plot_type, [])+[""]*(plotGroup+1)
            scatters_now = [scatters[ind] for ind, legend_name in enumerate(
                legend) if len(legend_name) > 0]
            legend_now = [
                legend_name for legend_name in legend if len(legend_name) > 0]
            if len(legend_now) > 0:
                legends_sort_tmp = legends_sort+[plotGroup+1]*(plotGroup+1)
                scatters_tmp = sorted(
                    scatters_now, key=lambda x: legends_sort[scatters_now.index(x)])
                legend_tmp = sorted(
                    legend_now, key=lambda x: legends_sort[legend_now.index(x)])
                plt.legend(scatters_tmp, legend_tmp, fontsize=15, framealpha=0,
                           handletextpad=0, markerscale=3, edgecolor="none")

        # set title
        title = kwargs.get("title")
        if isinstance(title, str):
            subplots[0].set_title(title)

        if valid_exportImagePath is not None:
            fig.savefig(valid_exportImagePath, dpi=300,
                        bbox_inches="tight", pad_inches=0.05)
            print(f"Figure is saved as {valid_exportImagePath}")
        return fig, subplots

    def plot_datas_fromXcms(self, xcms=[], title_func=None, flag_titleIsXcmpath=False, exportImagePath_func=None, **kwargs_in):
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
                kwargs_add["title"] = xcm_path

            if callable(exportImagePath_func):
                kwargs_add["exportImagePath"] = exportImagePath_func(xcm_path)

            kwargs_now = {**kwargs_in, **kwargs_add}
            fig, subplots = self.plot_datas(**kwargs_now)
            return_dic[xcm_path] = {"fig": fig, "subplots": subplots}
        return return_dic

    # # parameter

    def obtainParamsPerComp(self, model):
        comps = [model.__dict__[s] for s in model.componentNames]
        return sum([[{"param": comp.__dict__[s], "comp":comp} for s in comp.parameterNames] for comp in comps], [])

    def _obtainELF(self, error, frozen, link):
        error_str = "/".join(map(str, error))
        elf = "frozen" if frozen else (link if link != "" else error_str)
        return elf

    def obtainInfoParam(self, param, info_comp={}, info_model={}, flag_forPrint=False):
        ind_param_start = info_model.get("startParIndex", None)
        _ind_dict = {
            "ind_param": param.index,
            "ind_param_total": param.index+ind_param_start - 1 if ind_param_start is not None else "None",
            "ind_comp": info_comp.get("ind", "None"),
            "ind_model": info_model.get("ind", "None"),
        }
        _info_param_dict = {
            "ind": {
                "i_p": _ind_dict["ind_param"],
                "i_pT": _ind_dict["ind_param_total"],
                "i_c": _ind_dict["ind_comp"],
                "i_m": _ind_dict["ind_model"],
            } if flag_forPrint is True else _ind_dict,
            "main": {
                "name_comp": info_comp.get("name", "None"),
                "name_param": param.name,
                "unit": param.unit,
                "value": param.values[0]
            },
            "forNotPrint": {} if flag_forPrint is True else {
                "error": param.error[:2],
                "frozen": param.frozen,
                "link": param.link},
            "elf": {
                "elf": self._obtainELF(error=param.error[:2], frozen=param.frozen, link=param.link)
            }
        }
        return {**_info_param_dict["ind"], **_info_param_dict["main"], **_info_param_dict["forNotPrint"], **_info_param_dict["elf"]}

    def obtainInfoParamsAll(self, flag_forPrint=False, flag_oneOnly=False):
        AllModels = self.AllModels
        AllData = self.AllData

        nGroups = AllData.nGroups
        models = [AllModels(ind_model) for ind_model in range(1, nGroups+1)]

        _info_paramss = [[
            self.obtainInfoParam(
                paramAndComp["param"],
                info_comp={
                    "ind": model.componentNames.index(paramAndComp["comp"].name)+1,
                    "name":paramAndComp["comp"].name
                },
                info_model={
                    "startParIndex": model.startParIndex,
                    "ind": ind_model+1
                },
                flag_forPrint=flag_forPrint
            ) for paramAndComp in self.obtainParamsPerComp(model)] for ind_model, model in enumerate(models)]
        info_paramss = [_info_paramss[0]
                        ] if flag_oneOnly is True else _info_paramss
        return info_paramss

    def showParamsAll(self, flag_oneOnly=False, max_width=100):
        Fit = self.Fit
        chisq = Fit.statistic/Fit.dof

        info_paramss = self.obtainInfoParamsAll(
            flag_forPrint=True, flag_oneOnly=flag_oneOnly)

        # table
        table = texttable.Texttable(max_width=max_width)
        table.set_deco(texttable.Texttable.HEADER)
        table.set_cols_align(["r", "r", "r", "r", "l", "l", "l", "l", "l"])
        header = list(info_paramss[0][0].keys())
        table.add_rows([header]+sum(sum([[[
            [""]*(len(header))], [list(s.values()) for s in info_params]]
            for ind_model, info_params in enumerate(info_paramss)], []), []))
        print(table.draw()+f"\n\t\tReduced Chi-Squared: {chisq}")

    def showParamsAll_xspec(self, flag_oneOnly=False):
        AllModels = self.AllModels
        Fit = self.Fit

        if flag_oneOnly is True:
            AllModels(1).show()
        else:
            AllModels.show()
        Fit.show()

    def calcFluxLumin(self, cmdStr, flag_flux=True, flag_omit=True):
        AllModels = self.AllModels
        AllData = self.AllData
        nGroups = AllData.nGroups

        if flag_flux is True:
            AllModels.calcFlux(cmdStr)
        else:
            AllModels.calcLumin(cmdStr)
        output = [AllModels(ind_model).flux if flag_flux is True else AllModels(
            ind_model).lumin for ind_model in range(1, nGroups+1)]

        if flag_omit is True and all(s == output[0] for s in output):
            return [output[0]]

        return output
