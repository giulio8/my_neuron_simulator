import numpy as np
import matplotlib.pyplot as plt
import neuron


def myPlotFigure (common_axis:plt.Axes, xaxis, yaxis, title, unitLabel="", labels=None, nplots=1):
    if (nplots > 1):
        for i in range(nplots):
            if (labels != None):
                common_axis.plot(xaxis, yaxis[i], label=labels[i], linewidth="0.5")
            else:
                common_axis.plot(xaxis, yaxis[i], linewidth="0.5")
    else:
        common_axis.plot(xaxis, yaxis, linewidth="0.5")

    if (labels != None):
        common_axis.legend()
    common_axis.set_title(title)
    common_axis.set_xlabel("Time (s)")
    if (unitLabel != ""):
        common_axis.set_ylabel(unitLabel)


def drawFigure(simulation_time, time_step, variables: neuron.Variables, time_window=None, n_drawings=1, n_plots=1, axes=None, labels=None, dpixel=400):

    plt.rcParams["legend.loc"] = "upper right"

    rows = int(np.ceil(n_drawings/2))
    columns = 1 if n_drawings == 1 else 2
    figsize = (3, 2) if n_drawings == 1 else (10, 3*rows)
    start, end = time_window if time_window != None else (0, simulation_time)
    t_axis = np.arange(start, end, time_step)
    if (time_window != None):
        variables = variables.restrict(int(start/time_step), int(end/time_step)+1)
    fig, axes = plt.subplots(rows, columns, sharex=True, layout="constrained", figsize=figsize, dpi=dpixel)

    if (n_plots == 1):
        if (n_drawings > 1):
            for idx, (ax, (key, variable)) in enumerate(zip(axes.flat, variables)):
                myPlotFigure(ax, t_axis, variable.value, variable.title, variable.unit)
        else:
            el = variables.__iter__().__next__()[1]
            myPlotFigure(axes, t_axis, el.value, el.title, el.unit)
    else:
        if (n_drawings > 1):
            for idx, (ax, (key, variable)) in enumerate(zip(axes.flat, variables)):
                myPlotFigure(ax, t_axis, variable.value, variable.title, variable.unit, labels=labels, 
                             nplots = n_plots
                             )
        else:
            el = variables.__iter__().__next__()[1]
            myPlotFigure(axes, t_axis, el.value, el.title, el.unit, labels=labels, nplots = n_plots)

    plt.show()

    if (n_plots == 1):
        if (variables.S.array == True):
            print("Average number of APs: %.1f (%.1f Hz)"%(np.sum(variables.S.value), np.sum(variables.S.value)/simulation_time))
        if (variables.release_vector.array == True):
            print("Average number of vesicle releases: %.1f (%.1f Hz)"%(np.sum(variables.release_vector.value), np.sum(variables.release_vector.value)/simulation_time))
        if (variables.S.array == True):
            print("Average mutual information: %.4f bit/s"% (np.sum(variables.mutual_information.value)/(simulation_time/time_step))) # Bitrate already has s^(-1)
    else:
        if (variables.S.array == True):
            for i, value in enumerate(variables.S.value):
                print("Average number of APs: %.1f (%.1f Hz)"%(np.sum(value), np.sum(value)/simulation_time))
                print(labels[i])
        if (variables.release_vector.array == True):
            for i, value in enumerate(variables.release_vector.value):
                print("Average number of vesicle releases: %.1f (%.1f Hz)"%(np.sum(value), np.sum(value)/simulation_time))
                print(labels[i])
        if (variables.S.array == True):
            for i, value in enumerate(variables.mutual_information.value):
                print("Average mutual information: %.4f bit/s"% (np.sum(value)/(simulation_time/time_step))) # Bitrate already has s^(-1)
                print(labels[i])