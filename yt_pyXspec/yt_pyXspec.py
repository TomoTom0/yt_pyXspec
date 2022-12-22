## import
import os, re, sys, shutil, glob2, xspec
import numpy as np
import matplotlib.pyplot as _plt
from matplotlib import gridspec

class ytpx():
    ## init
    def __init__(self, dir_path=None):
        if os.path.isdir(dir_path):
            print(f"change directory to {dir_path}")
            os.chdir(dir_path)
        self.findXcm(dir_path)
        
        self.Plot=xspec.Plot
        self.AllModels=xspec.AllModels
        self.AllData=xspec.AllData
        self.Xset=xspec.Xset
        self.Fit=xspec.Fit
        self.plt=_plt
        self.xspec=xspec
        self.default_exportImagePath=None
        self.__initMatplotlibRcParams()
        self.__initXspecPlot()
        #self.gridspec=_gridspec
    def findXcm(self, dir_path=None, keyword=None):
        valid_dir_path=dir_path or self.dir_path
        valid_keyword=keyword or  "*.xcm"
        xcms=glob2.glob(valid_dir_path+"/"+valid_keyword) if valid_dir_path is not None else []
        self.xcms=xcms
        return xcms
    # Disable
    def blockPrint(self):
        sys.stdout = open(os.devnull, "w")
    # Restore
    def enablePrint(self):
        sys.stdout = sys.__stdout__

    def __initXspecPlot(self):
        Plot=self.Plot
        Plot.xAxis="keV"
        Plot.addCommand("time off")
        Plot.addCommand("font roman")

        Plot.addCommand("cs 2")
        Plot.addCommand("lw 5")
        Plot.addCommand("win 1")
        Plot.addCommand("loc 0.05 0.05 1 0.95")
        Plot.addCommand("win 2")
        Plot.addCommand("loc 0.05 0.05 1 0.95")

    def __initMatplotlibRcParams(self):
        plt=self.plt
        plt.rcParams["figure.figsize"] = (3.14*2, 3.14*2*0.7)
        plt.rcParams["figure.dpi"] = 100 # 画像保存するときは300に
        
        plt.rcParams["font.family"] = "Times New Roman"      #全体のフォントを設定
        plt.rcParams["xtick.direction"] = "in"               #x軸の目盛線を内向きへ
        plt.rcParams["ytick.direction"] = "in"               #y軸の目盛線を内向きへ
        plt.rcParams["xtick.minor.visible"] = True           #x軸補助目盛りの追加
        plt.rcParams["ytick.minor.visible"] = True           #y軸補助目盛りの追加
        plt.rcParams["xtick.major.width"] = 1.5              #x軸主目盛り線の線幅
        plt.rcParams["ytick.major.width"] = 1.5              #y軸主目盛り線の線幅
        plt.rcParams["xtick.minor.width"] = 1.0              #x軸補助目盛り線の線幅
        plt.rcParams["ytick.minor.width"] = 1.0              #y軸補助目盛り線の線幅
        plt.rcParams["xtick.major.size"] = 10                #x軸主目盛り線の長さ
        plt.rcParams["ytick.major.size"] = 10                #y軸主目盛り線の長さ
        plt.rcParams["xtick.minor.size"] = 5                 #x軸補助目盛り線の長さ
        plt.rcParams["ytick.minor.size"] = 5                 #y軸補助目盛り線の長さ
        plt.rcParams["font.size"] = 14                       #フォントの大きさ
        plt.rcParams["axes.linewidth"] = 1.5                 #囲みの太さ

    def loadXcm(self, xcm_path):
        if not os.path.isfile(xcm_path):
            return False
        self.blockPrint()
        self.Xset.restore(xcm_path)
        self.enablePrint()
        return True
    #def ___load(self, xcm_content):
    #    pass
    
    ## obtain_xyss
    def obtain_xyss_s_fromXcms(self, xcms=[], **kwargs_in):
        # plots
        xyss_dic={}
        for xcm_path in xcms:
            self.loadXcm(xcm_path)
            xyss_now=self.obtain_xyss(**kwargs_in)
            xyss_dic[xcm_path]=xyss_now
        return xyss_dic
            
    def obtain_xyss(self, plots=["eeu"]):
        Plot=self.Plot
        AllData=self.AllData
        plot_command = " ".join(plots)
        Plot(plot_command)
        xyss_s = {}

        for plotWindow_tmp, plot_type in enumerate(plots):
            plotWindow = plotWindow_tmp+1
            xyss = {}
            xs, ys, xe, ye, ys_model, ys_comps = [[]]*6
            for plotGroup in range(1, AllData.nGroups+1):
                xs = Plot.x(plotGroup, plotWindow)
                if plot_type in {"eeu", "ld" ,"ratio", "del"}:
                    ys = Plot.y(plotGroup, plotWindow)
                    xe = Plot.xErr(plotGroup, plotWindow)
                    ye = Plot.yErr(plotGroup, plotWindow)

                if plot_type in {"eeu", "ld" ,"eem"}:
                    ys_model = Plot.model(plotGroup, plotWindow)

                # obtain comps in models
                ys_comps = []
                comp_N = 1
                while True:
                    try:
                        ys_tmp = Plot.addComp(comp_N, plotGroup, plotWindow)
                        comp_N += 1
                        # execlude components with only 0
                        if sum([1 for s in ys_tmp if s == 0]) == len(ys_tmp):
                            continue
                        ys_comps.append(ys_tmp)
                    except:
                        break
                xyss[plotGroup] = {"xs": xs, "ys": ys, "xe": xe,
                                "ye": ye, "ys_model": ys_model, "ys_comps": ys_comps}
            xyss_s[plot_type]=xyss

        return xyss_s

    def obtain_xyss_s_dic(self, xcm_paths_dic, plots_dic):
        Xset=self.Xset
        xyss_s_dic={}
        for key_xcms, xcm_paths in xcm_paths_dic.items():
            xyss_ss={}
            for key_xcm, xcm_path in xcm_paths.items():
                self.loadXcm(xcm_path)
                xyss_ss[key_xcm]=self.obtain_xyss(plots=plots_dic[key_xcms])

            xyss_s_dic[key_xcms]=self.combine_xyss_ss(xyss_ss=xyss_ss)
        return xyss_s_dic

    def combine_xyss_ss(self, xyss_ss={}, sortKeys=[]):
        xyss_s_new={}
        dataIndexs={}
        if set(sortKeys) == set(xyss_ss.keys()):
            xyss_ss_list=[xyss_ss[k] for k in sortKeys]
        else:
            xyss_ss_list=list(xyss_ss.values())
        for xyss_s in xyss_ss_list:
            for plotName, xyss_dicts in xyss_s.items():
                if plotName not in xyss_s_new.keys():
                    dataIndexs[plotName]=1
                    xyss_s_new[plotName]={}
                for xyss_dict in xyss_dicts.values():
                    xyss_s_new[plotName][dataIndexs[plotName]]=xyss_dict
                    dataIndexs[plotName]+=1
        return xyss_s_new

    ## sub functions for plot
    def __checkWritableFilePath(self, path):
        if path is None:
            return False
        isFile_OK = (os.path.isfile(path) and os.access(path, os.W_OK))
        isNotExist_OK = (os.path.isdir(os.path.split(path)[0]) and not os.path.exists(path))
        return isFile_OK or isNotExist_OK

    def __extractValues(self, xyss):
        sum_dic={}
        for key_xy in ["x", "y"]:
            valss=[s.get(key_xy+"s", []) for s in xyss.values()]
            errss=[s.get(key_xy+"e", []) for s in xyss.values()]
            models=[s.get(key_xy+"s_model", []) for s in xyss.values()]
            sum_tmp=[]
            for vals, errs in zip(valss, errss):
                if len(errs)== len(vals):
                    sum_tmp+=list(np.array(vals*2)+np.array(errs*2)*np.array([1]*len(errs)+[-1]*len(errs)))
                else:
                    sum_tmp+=vals
            sum_dic[key_xy+"s"]=sum_tmp+sum(models, [])
            #xs_sum=sum([np.array(s["xs"]*2)+np.array(s["xe"]*2)*np.array([1]*len(s["xe"])+[-1]*len(s["xe"])) for s in xys["eeu"].values()], [])
            #ys_sum=sum([np.array(s["ys"]*2)+np.array(s["ye"]*2)*np.array([1]*len(s["ye"])+[-1]*len(s["ye"])) for s in xys["eeu"].values()], [])
        return sum_dic#{"xs":xs_sum, "ys":ys_sum}
    
    def __obtainLim(self, values, logIsValid=True, margin_ratio=0.05):
        v_min=min(values)
        v_max=max(values)
        if logIsValid is True:
            margin=(np.log10(v_max)-np.log10(v_min))*0.05
            return (10**(np.log10(np.array([v_min,v_max]))+np.array([-1,+1])*margin)).tolist()
        else:
            margin=(v_max-v_min)*0.05
            return ((np.array([v_min,v_max]))+np.array([-1,+1])*margin).tolist()

    ## plot xyss
    def plot_xyss(self,
        plots=["eeu"],
        x_lim=[],
        y_lims={},     
        colors=["royalblue", "red", "olivedrab", "turqoise", "orange", "chartreuse", "navy" ,"firebrick", "darkgreen", "darkmagenta"], 
        markers=[],
        legends_dic={},
        legends_sort=[],
        xyss_s=None,
        exportImagePath=None, **kwargs_in):

        # kwargs
        kwargs_default={
            "flag_dataPlot":True,
            "flag_modelPlot":True,
            "flag_compPlot":True,
            "flag_modelStep":False,
            "marker_size_data":0,
            "elinewith_data":1,
            "marker_size_mdoel":0,
            "elinewidth_model":0.5,
            "title":None
        }
        kwargs={**kwargs_default, **kwargs_in}

           
        plt=self.plt
        default_exportImagePath=self.default_exportImagePath
        obtain_xyss=self.obtain_xyss
        checkWritableFilePath=self.__checkWritableFilePath
        valid_exportImagePath=([s for s in [exportImagePath, default_exportImagePath] if checkWritableFilePath(s)]+[None])[0]

        plots_diff = set(plots)-{"eeu", "eem", "ratio", "del", "ld"}
        if len(plots_diff) >= 1:
            print(", ".join(list(plots_diff))+" are not appropriate")

        fig = plt.figure()
        fig.patch.set_facecolor("white")
        xyss_s_valid = xyss_s or obtain_xyss(plots=plots)
            
        subplots = []

        # set height ratios for sublots
        gs = gridspec.GridSpec(len(plots), 1, height_ratios=[
                            2 if s in ["eeu", "eem", "ld"] else 1 for s in plots])

        for gs_tmp, plot_type in zip(gs, plots):
            xyss = xyss_s_valid.get(plot_type, None)
            if xyss is None:
                continue
            
            # the fisrt subplot
            ax = plt.subplot(gs_tmp)
            subplots.append(ax)
                        
            #### set scale
            ax.set_xscale("log")
            
            if not plot_type in ["ratio", "del"]:
                ax.set_yscale("log")
            elif plot_type in ["ratio"]:
                plt.axhline(1,ls=":", lw = 1., color="black", alpha=1, zorder=0)
            elif plot_type in ["del"]:
                plt.axhline(0,ls=":", lw = 1., color="black", alpha=1, zorder=0)

            #### set lim
            xys_sum=self.__extractValues(xyss)
            scaleIsLog= {
                "x":ax.get_xscale()=="log",
                "y":ax.get_yscale()=="log"}
            lims_fromValue={key_xy+"s":self.__obtainLim(xys_sum[key_xy+"s"], logIsValid=scaleIsLog.get(key_xy, True))
                            for key_xy in ["x", "y"]}
            ax.set_xlim(lims_fromValue["xs"])
            ax.set_ylim(lims_fromValue["ys"])

            y_lim=y_lims.get(plot_type, None)
            if hasattr(x_lim, "__iter__") and not len(x_lim)<2:
                ax.set_xlim(x_lim)
            if hasattr(y_lim, "__iter__") and not len(y_lim)<2:
                ax.set_ylim(y_lim)
                    
            #### set label
            plt.subplots_adjust(hspace=.0)
            if not gs_tmp == gs[-1]:
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_xminorticklabels(), visible=False)
            else:
                plt.xlabel("Energy (keV)")
            ylabel_dic = {"eeu": r"keV$\mathdefault{^2}$ (Photons cm$^\mathdefault{-2}$ s$^\mathdefault{-1}$ keV$^\mathdefault{-1}$)",
                        "eem": "$EF_\mathdefault{E}$ (keV$^2$)",
                        "ld": "normalized counts s$^\mathdefault{-1}$ keV$^\mathdefault{-1}$",
                        "ratio": "ratio", "del": "residual"}
            plt.ylabel(ylabel_dic[plot_type], fontsize=10)

            fig.align_labels()

            # plot eeu
            scatters=[]
            for plotGroup, xys_tmp in xyss.items():
                xs = xys_tmp["xs"]
                marker=(markers+["o"]*(plotGroup+1))[plotGroup-1]
                color=(colors+["black"]*(plotGroup+1))[plotGroup-1]
                if plot_type in {"eeu", "eem" ,"ratio", "del", "ld"} and kwargs.get("flag_dataPlot") is True:
                    ys = xys_tmp["ys"]
                    xe = xys_tmp["xe"]
                    ye = xys_tmp["ye"]
                    marker_size=kwargs.get("marker_size_data", 0)
                    elinewith=kwargs.get("elinewidth_data", 1)
                    scatter_tmp=plt.scatter(xs,ys, marker=marker, color=color, s=marker_size)
                    scatters.append(scatter_tmp)
                    plt.errorbar(xs, ys, yerr=ye, xerr=xe, capsize=0, fmt=marker, markersize=0,
                                ecolor=color, markeredgecolor="none", color="none", elinewidth=elinewith)

                if plot_type in {"eeu", "eem", "ld"} and kwargs.get("flag_modelPlot") is True:
                    ys_model = xys_tmp["ys_model"]
                    #plt.plot(xs, ys_model, color=color)
                    #plt.scatter(xs, ys_model, color=color, marker="_")
                    xems=sorted(zip(xs,xe,ys_model), key=lambda x:x[0])
                    xs_tmp=[s[0] for s in xems]
                    xe_tmp=[s[1] for s in xems]
                    ys_tmp=[s[2] for s in xems]

                    xs_step=[x-xe_tmp for x, xe_tmp in zip(xs_tmp,xe_tmp)]+[xs_tmp[-1]+xe_tmp[-1]]
                    ys_step=[ys_tmp[0]]+ys_tmp
                    
                    marker_size=kwargs.get("marker_size_model", 0)
                    elinewith=kwargs.get("elinewidth_model", 0.5)


                    if kwargs.get("flag_modelStep") is True:
                        plt.step(xs_step, ys_step, color=color, linewidth=0.5, solid_joinstyle="miter", markeredgecolor="none")
                    else:
                        plt.scatter(xs, ys_model, color=color, marker="_", s=marker_size)
                        plt.errorbar(xs, ys_model , xerr=xe, capsize=0, fmt=marker, markersize=0,
                                ecolor=color, markeredgecolor="none", color="none", elinewidth=elinewith)
                
                if kwargs.get("flag_compPlot") is True:
                    ys_comps = xys_tmp["ys_comps"]
                    for ys_comp in ys_comps:
                        plt.plot(xs, ys_comp, linestyle="dotted",
                                color=color)
                # plt.plot()
            legend=legends_dic.get(plot_type, [])+[""]*(plotGroup+1)
            scatters_now=[scatters[ind] for ind, legend_name in enumerate(legend) if len(legend_name)>0]
            legend_now=[legend_name for legend_name in legend if len(legend_name)>0]
            if len(legend_now)>0:
                legends_sort_tmp=legends_sort+[plotGroup+1]*(plotGroup+1)
                scatters_tmp=sorted(scatters_now, key=lambda x:legends_sort[scatters_now.index(x)])
                legend_tmp=sorted(legend_now, key=lambda x:legends_sort[legend_now.index(x)])
                plt.legend(scatters_tmp, legend_tmp, fontsize=15, framealpha=0, handletextpad=0,markerscale=3, edgecolor="none")
        
        #### set title
        title=kwargs.get("title")
        if isinstance(title, str):
            subplots[0].set_title(title)
    
        if valid_exportImagePath is not None:
            fig.savefig(valid_exportImagePath, dpi=300, bbox_inches="tight", pad_inches=0.05)
            print(f"Figure is saved as {valid_exportImagePath}")
        return fig, subplots
    
    def plot_xyss_fromXcms(self, xcms=[], title_func=None, flag_titleIsXcmpath=False, exportImagePath_func=None, **kwargs_in):
        return_dic={}
        for xcm_path in xcms:
            loadSuccess=self.loadXcm(xcm_path=xcm_path)
            if loadSuccess is False:
                print(f"Failed to load {xcm_path}")
                continue
            
            kwargs_add={}
            if callable(title_func):
                kwargs_add["title"]=title_func(xcm_path)
            elif flag_titleIsXcmpath is True:
                kwargs_add["title"]=xcm_path
            
            if callable(exportImagePath_func):
                kwargs_add["exportImagePath"]=exportImagePath_func(xcm_path)
                    
            kwargs_now={**kwargs_in, **kwargs_add}
            fig, subplots=self.plot_xyss(**kwargs_now)
            return_dic[xcm_path]={"fig":fig, "subplots":subplots}
        return return_dic
            