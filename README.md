# my_neuron_simulator
Light tripartite synapse simulator

This software is organized in few documents, each functionality is described here.

<ul>
  <li><b>neuron.py</b> is the core of the simulator: provides an object oriented implementation of the synapse, with several adjustable parameters to modify the biophysical properties of the cell (membrane, AP, vesicle release time constants and many more) and the stimulation parameters (injected current intensity, frequency, etc). It also has a specific attribute for noise characteristic of the synapse, making easy to toggle each contribute to study the effects. Current version implements a simple but functional trick to manage variables calculation and storage dynamically, based on which of them the user wants to plot (and so, to store necessairly in an array), and which doesn't. </li>
  <li><b>lib.py</b> is the scientific library: it's a list of functions that implement the equations that governs the neuron. Neuron class call these methods and it's independent from their implementation. A numerical method has been preferred to mimic the differential equations, so a very small time interval is required to obtain sensible results.</li>
  <li><b>myPlot.py</b> is a simple api that calls the matplotlib module and automathically print a single image with all the plots, and adjust its behavior if the user wants to plot multiple lines in a single plot (e.g. in comparisons)</li>
  <li><b>main.ipynb</b> is the external caller of our simulator, here, to enjoy our simulation results, we just have to:
    <ul>
      <li>Set Properties object containing biophysical properties of the neuron we want to simulate</li>
      <li>Set SimulationParameters object containing our simulation parameters (e.g. fixed AP rate or dynamical stimulation via injected current in the membrane, number of iterations to simulate)</li>
      <li>Set Noise object containing the noise components that we want to include (e.g. escape noise vs fixed threshold, axonal noise, etc)</li>
      <li>Create a Neuron instance passing the previously created objects</li>
      <li>Call simulate method and the myPlot module to show results</li>
    </ul>
  <li><b>processInspector.ipynb</b> is a detatched document useful during long sessions of simulation. Executing it in a separate process from the main simulation, it can periodically show the partial results and plot them, without requiring a significant amount of resources slowing down the main process. This way, we don't have to wait till the end of the simulation to check the correctness of the results.</li>
  </ul
