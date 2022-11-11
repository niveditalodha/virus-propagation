# Virus Propagation

Team:
Vamshi Chidara (schidara), Akhil Kumar Mengani (amengan), Nivedita Lodha(nnlodha)

Project Details:
This project is an efficient implementation of an algorithm to determine whether a virus will result in an epidemic or die quickly in a static network.

Python program file: virus_propagation.py<br>
Purpose: The algorithm in the file helps us calculate the effective strength of the virus, the simulation of the propagation of the virus in the network and the implementation of 4 different immunization policies to prevent the spread of the virus in the network. The program also finds the minimum number of vaccines needed in each policy to reduce the strength of virus below the threshold 1.

1. Software Requirements:
OS - Windows 11
Python 3.8.2

Libraries:
sys : comes default with python<br>
operator: comes default with python<br>
networkx - 2.8.7 : to install networks, run "pip install networkx" in the terminal<br>
numpy - 1.18.4 : to install numpy, run "pip install numpy" in the terminal<br>
matplotlib - 3.4.3 : to install matplotlib, run "pip install matplotlib" in the terminal<br>

2. Environment Variables: None

3. Instructions To Run:
Create an empty output folder in the project directory <br>
In the main project directory, <br>
Run Command: python <file.py><br>
In this case:  python virus_propagation.py<br>

4. Input and Output Files:
The input file is a static.network graph file where each line represents nodes that have edges between them

After running the command as given in 3. Instructions To Run , outputs are written to the output folder. An output file with the name as description of the graph is present in the output folder.<br>
We receive the values of effective strength in the terminal for the respective alpha and beta values:<br>
Example:<br>
For beta = 0.2 and delta = 0.7, effective strengh = 12.529913074497653<br>
.....<br>
Immunization using Policy A, k=200<br>
For beta = 0.2 and delta = 0.7, effective strengh = 12.308842563118665<br>
For beta = 0.01 and delta = 0.6, effective strengh = 0.718015816181922<br>
.....<br>
