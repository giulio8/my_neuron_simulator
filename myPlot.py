import numpy as np
import matplotlib.pyplot as plt
import neuron


def myPlotFigure (common_axis:plt.Axes, xaxis, yaxis, title, unitLabel="", labels=None, nplots=1):
    if (nplots > 1):
        for i in range(nplots):
            if (labels != None):
                y = yaxis[i] if len(yaxis.shape) == 1 else yaxis[i, :]
                common_axis.plot(xaxis, y, label=labels[i], linewidth="0.5")
            else:
                y = yaxis[i] if len(yaxis.shape) == 1 else yaxis[i, :]
                common_axis.plot(xaxis, y, linewidth="0.5")
    else:
        y = np.reshape(yaxis, (yaxis.size,))
        common_axis.plot(xaxis, y, linewidth="0.5")

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
        idx_start = int(np.ceil(start/time_step))
        idx_end = idx_start + len(t_axis)
        variables_to_plot = variables.restrict(idx_start, idx_end)
    else:
        variables_to_plot = variables
    fig, axes = plt.subplots(rows, columns, sharex=True, layout="constrained", figsize=figsize, dpi=dpixel)

    if (n_plots == 1):
        if (n_drawings > 1):
            for ax, (key, variable) in zip(axes.flat, variables_to_plot):
                myPlotFigure(ax, t_axis, variable.value, variable.title, variable.unit)
        else:
            el = variables_to_plot.__iter__().__next__()[1]
            myPlotFigure(axes, t_axis, el.value, el.title, el.unit)
    else:
        if (n_drawings > 1):
            for ax, (key, variable) in zip(axes.flat, variables_to_plot):
                myPlotFigure(ax, t_axis, variable.value, variable.title, variable.unit, labels=labels, 
                             nplots = n_plots
                             )
        else:
            el = variables_to_plot.__iter__().__next__()[1]
            myPlotFigure(axes, t_axis, el.value, el.title, el.unit, labels=labels, nplots = n_plots)

    plt.show()

    if (n_plots == 1):
        if (variables_to_plot.S.array == True):
            print("Average number of APs: %.1f (%.1f Hz)"%(np.sum(variables_to_plot.S.value), np.sum(variables_to_plot.S.value)/simulation_time))
        if (variables_to_plot.release_vector.array == True):
            print("Average number of vesicle releases: %.1f (%.1f Hz)"%(np.sum(variables_to_plot.release_vector.value), np.sum(variables_to_plot.release_vector.value)/simulation_time))
        if (variables_to_plot.S.array == True):
            print("Average mutual information: %.4f bit/s"% (np.sum(variables_to_plot.mutual_information.value)/(simulation_time/time_step))) # Bitrate already has s^(-1)
    else:
        if (variables_to_plot.S.array == True):
            for i, value in enumerate(variables_to_plot.S.value):
                print("Average number of APs: %.1f (%.1f Hz)"%(np.sum(value), np.sum(value)/simulation_time))
                print(labels[i])
        if (variables_to_plot.release_vector.array == True):
            for i, value in enumerate(variables_to_plot.release_vector.value):
                print("Average number of vesicle releases: %.1f (%.1f Hz)"%(np.sum(value), np.sum(value)/simulation_time))
                print(labels[i])
        if (variables_to_plot.S.array == True):
            for i, value in enumerate(variables_to_plot.mutual_information.value):
                print("Average mutual information: %.4f bit/s"% (np.sum(value)/(simulation_time/time_step))) # Bitrate already has s^(-1)
                print(labels[i])