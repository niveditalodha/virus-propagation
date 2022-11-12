import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from copy import copy, deepcopy
from operator import itemgetter

def read_graph():
    graph = nx.Graph()
    filename = 'static.network'
    with open(filename) as file:
        # skip header line
        next(file)
        for line in file:
            node1, node2 = int(line.split()[0]), int(line.split()[1])
            graph.add_edge(node1, node2)
    return graph

def calculate_beta_plot_delta(max_eigen_val, delta):
    delta_values = list()
    minimum_beta = 0
    count = 0
    for i in range(1, 1001):
        b = i/1000
        cvpm = b/delta
        strngth = max_eigen_val*cvpm
        # get the minimum beta to cause an epidemic
        if count == 0 and strngth > 1:
            minimum_beta = b
            count = 1
        delta_values.append(strngth)
    print("Minimum transmission probability beta = {} for delta = {}".format(minimum_beta, delta))
    plot_betaVSdelta(delta_values, 'Beta',
                'strength vs different beta values for delta = {}'.format(delta))


def calculate_delta_plot_beta(max_eigen_val, beta):
    beta_values = list()
    maximum_delta = 1
    count = 0
    for i in range(1, 1001):
        d = float(i)/1000
        cvpm = beta/d
        strngth = max_eigen_val*cvpm
        # get the maximum delta to cause an epidemic
        if strngth < 1 and count == 0:
            maximum_delta = d
            count = 1
        beta_values.append(strngth)
    print("Maximum healing probability delta = {} for beta = {}".format(maximum_delta, beta))
    plot_betaVSdelta(beta_values, 'beta',
                'strength vs different delta values for beta = {}'.format(beta))


def plot_betaVSdelta(strength, parameter, title):
    parameter_list = np.arange(0.001, 1.001, 0.001)
    plt.plot(parameter_list, strength)
    plt.title(title)
    plt.xlabel(parameter)
    plt.ylabel('Effective Strength')
    plt.axhline(y=1, linewidth=2, color='r')
    plt.savefig('output/{}.png'.format(title), bbox_inches='tight')
    plt.close()


def simulation(beta, delta, matrix, number_of_nodes, policy_type=""):
    plot_series = list()
    for i in range(10):
        random_infected_population = list()
        count = number_of_nodes//10
        infected_bool = list()
        for i in range(number_of_nodes):
            infected_bool.append(0)
        while len(random_infected_population) < count:
            r = random.randint(0, number_of_nodes-1)
            if not infected_bool[r]:
                infected_bool[r] = 1
                random_infected_population.append(r)

        number_of_infected_population = [len(random_infected_population)]

        for i in range(100):
            current_infected = set()
            current_recovered = list()
            for infected in random_infected_population:
                for j in range(len(matrix[infected])):
                    if matrix[infected][j] == 1:
                        r = random.randint(1, 10)/10 
                        if r < beta:
                            current_infected.add(j)
                r = random.randint(1, 10)/10
                if r >= delta:
                    current_infected.add(infected)
                else:
                    current_recovered.append(infected)

            for node in current_recovered:
                infected_bool[node] = 0
            for node in current_infected:
                infected_bool[node] = 1
            number_of_infected_population.append(len(current_infected))
            random_infected_population = current_infected
        plot_series.append(number_of_infected_population)
    plot_simulation(plot_series, number_of_nodes, '{} Simulation with beta = {} and delta = {}'.format(
        policy_type, beta, delta))
    return number_of_infected_population


def plot_simulation(plot_series, number_of_nodes, title):
    plot_series = np.array(plot_series) / number_of_nodes
    plot_series = np.mean(plot_series, axis=0)
    plt.plot(plot_series)
    plt.title(title)
    plt.xlabel('times infected')
    plt.ylabel('average ratio of infected nodes')
    plt.savefig('output/{}.png'.format(title), bbox_inches='tight')
    plt.close()


def calculate_effective_strength(adj):
    eig_val, eig_vec = np.linalg.eig(adj)
    eig_set = [(eig_val[i], eig_vec[i]) for i in range(len(eig_vec))]
    eig_set = sorted(eig_set, key=lambda x: x[0], reverse=1)
    max_val = eig_set[0][0].real
    s1 = max_val * cvpm1
    s2 = max_val * cvpm2
    print("For beta = {} and delta = {}, effective strength = {}".format(
        beta1, delta1, s1))
    print("For beta = {} and delta = {}, effective strength = {}".format(
        beta2, delta2, s2))
    return max_val


def policyA(matrix_policyA, number_of_nodes, simulate = False, k=200):
    print("\nImmunization using Policy A, k={}".format(k))
    immunized = set()
    while len(immunized) < k:
        r = random.randint(0, number_of_nodes-1)
        if r not in immunized:
            immunized.add(r)

    for node in immunized:
        for i in range(len(matrix_policyA[node])):
            matrix_policyA[node][i] = 0
            matrix_policyA[i][node] = 0
    
    calculate_effective_strength(matrix_policyA)
    if simulate:
        print("Simulation for Policy A")
        simulation(beta1, delta1, matrix_policyA, number_of_nodes, 'Policy A')
    return matrix_policyA

def policyB(matrix_policyB, g, simulate=False, k=200):
    print("\nImmunization using Policy B, k={}".format(k))
    immunized = set()
    degree = sorted(list(g.degree()), key=lambda x: x[1], reverse=1)
    for i in range(k):
        immunized.add(degree[i][0])

    for node in immunized:
        for i in range(len(matrix_policyB[node])):
            matrix_policyB[node][i] = 0
            matrix_policyB[i][node] = 0
    calculate_effective_strength(matrix_policyB)
    if simulate:
        print("Simulation for Policy B")
        simulation(beta1, delta1, matrix_policyB, nx.number_of_nodes(g), 'Policy B')
    return matrix_policyB


def policyC(g, simulate=False, k=200):
    print("\nImmunization using Policy C, k={}".format(k))
    immunized = set()
    number_of_nodes = nx.number_of_nodes(g)
    while len(immunized) < k:
        degree = sorted(list(g.degree()), key=lambda x: x[1], reverse=1)
        immunized.add(degree[0][0])
        g.remove_node(degree[0][0])

    matrix_policyC = [[0 for i in range(number_of_nodes)] for j in range(number_of_nodes)]
    for i in nx.edges(g):
        matrix_policyC[i[0]][i[1]] = 1
        matrix_policyC[i[1]][i[0]] = 1
    calculate_effective_strength(matrix_policyC)
    if simulate:
        print("Simulation for Policy C")
        simulation(beta1, delta1, matrix_policyC, number_of_nodes, 'Policy C')
    return matrix_policyC

def policyD(matrixD, g, simulate = False, k=200):
    print("\nImmunization using Policy D, k={}".format(k))
    number_of_nodes = nx.number_of_nodes(g)
    eigen_value, eigen_vector = np.linalg.eig(matrixD)
    eigen_set = [(eigen_value[i], eigen_vector[i]) for i in range(len(eigen_value))]
    eigen_set = sorted(eigen_set, key=lambda x: x[0], reverse=1)
    maximum_vector = eigen_set[0][1]
    maximum_vector = np.absolute(maximum_vector)
    target = [u[0] for u in sorted(enumerate(maximum_vector), reverse=True, key=itemgetter(1))[:k]]
    for i in target:
        g.remove_node(i)
    matrix_policyD = [[0 for i in range(number_of_nodes)] for j in range(number_of_nodes)]
    for i in nx.edges(g):
        matrix_policyD[i[0]][i[1]] = 1
        matrix_policyD[i[1]][i[0]] = 1
    calculate_effective_strength(matrix_policyD)
    if simulate:
        print("Simulation for Policy D")
        simulation(beta1, delta1, matrix_policyD, number_of_nodes, 'Policy D')
    return matrix_policyD


def calculate_minimum_vaccines(matrix, graph, policy_type):
    print("Calculating minimum number of vaccines for policy {}".format(policy_type))
    strengths = list()
    number_vaccines = list()
    flag = 1
    while flag:
        number_vaccines.append(100 if not number_vaccines else number_vaccines[-1] + 100)
        k = number_vaccines[-1]
        if policy_type == 'A':
            updated_matrix = policyA(matrix.copy(), nx.number_of_nodes(graph), k=k)
        elif policy_type == 'B':
            updated_matrix = policyB(matrix.copy(), graph.copy(), k=k)
        elif policy_type == 'C':
            updated_matrix = policyC(graph.copy(), k=k)
        elif policy_type == 'D':
            updated_matrix = policyD(matrix.copy(), graph.copy(), k=k)
        eigen_value, eigen_vector = np.linalg.eig(updated_matrix)
        eigen_set = [(eigen_value[i], eigen_vector[i]) for i in range(len(eigen_value))]
        eigen_set = sorted(eigen_set, key=lambda x: x[0], reverse=1)
        max_val = eigen_set[0][0].real
        strength = max_val * cvpm1
        if strength < 1:
            flag = 0
        strengths.append(strength)
    plot_minimum_vaccines(number_vaccines, strengths, 'Number of vaccines for policy {}'.format(policy_type))


def plot_minimum_vaccines(number_k, strengths, title):
    plt.plot(number_k, strengths)
    plt.title(title)
    plt.xlabel('number of vaccines')
    plt.ylabel('effective Strength')
    plt.axhline(y=1, linewidth=2, color='r')
    plt.savefig('output/{}.png'.format(title), bbox_inches='tight')
    plt.close()


def main():
    # Read data from graph file and create a networkx graph
    graph = read_graph()

    # Calculating the effective strength for SIS virus propagation model
    no_of_nodes = nx.number_of_nodes(graph)
    matrix = [[0 for i in range(no_of_nodes)] for j in range(no_of_nodes)]
    for i in nx.edges(graph):
        matrix[i[1]][i[0]], matrix[i[0]][i[1]] = 1, 1 #adjacency matrix for edges

    # Calculate infection spread across the network
    max_eigen_val = calculate_effective_strength(matrix)

    # Setting values of delta and finding beta that affect the effective strength
    # Plotting beta vs delta1
    calculate_beta_plot_delta(max_eigen_val, delta1)
    # Plotting beta vs delta2
    calculate_beta_plot_delta(max_eigen_val, delta2)

    # Setting values of beta and finding delta that affect the effective strength
    # Plotting delta vs beta1
    calculate_delta_plot_beta(max_eigen_val, beta1)
    # Plotting delta vs beta1
    calculate_delta_plot_beta(max_eigen_val, beta2)

    # Simulating virus propagation with SIS virus propagation model for beta 1 and delta 1
    simulation(beta1, delta1, matrix, no_of_nodes)

    # Simulating virus propagation with SIS virus propagation model for beta 2 and delta 2
    simulation(beta2, delta2, matrix, no_of_nodes)

    # Immunization using all policies and calculating its effective strength and its simulation
    policyA(matrix.copy(), no_of_nodes, True)
    policyB(matrix.copy(), graph.copy(), True)
    policyC(graph.copy(), True)
    policyD(matrix.copy(), graph.copy(), True)

    # Calculate minimum number of vaccines for all policies
    calculate_minimum_vaccines(matrix, graph, 'A')
    calculate_minimum_vaccines(matrix, graph, 'B')
    calculate_minimum_vaccines(matrix, graph, 'C')
    calculate_minimum_vaccines(matrix, graph, 'D')

random.seed(123)

# setting constant values
beta1 = 0.2
beta2 = 0.01
delta1 = 0.7
delta2 = 0.6
cvpm1 = beta1 / delta1
cvpm2 = beta2 / delta2

if __name__ == "__main__":
    main()