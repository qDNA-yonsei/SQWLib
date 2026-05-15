import numpy as np

def calculate_energys(n,J=1):
    
    N = 2**n
    
    energy = np.zeros(N)
    
    for state in range(N):
        
        binary = format(state, f"0{n}b")
        
        spins = [-1 if spin == '0' else 1 for spin in binary]
        
        for i in range(n-1):
            energy[state] += J*spins[i]*spins[i+1]
    
    return energy


def calculate_transition_list(energy,beta):
    
    transition_list = []
    
    N = len(energy)
    
    n = int(np.log2(N))
    
    for node in range(N):
        
        node_probability = 0
        
        for spin in range(n):
            
            neighbor = list(format(node, f'0{n}b'))
            
            neighbor[spin] = str(int(neighbor[spin])^1)
            
            neighbor = ''.join(neighbor)
            
            neighbor = int(neighbor,2)
            
            energy_diff = energy[node] - energy[neighbor]
            
            probability = min(1,np.exp(beta*energy_diff)) / n
            
            transition_list.append([node,neighbor,float(probability)])
            
            node_probability += probability
        
        probability = 1 - node_probability
        
        if probability < 0: probability = 0
        
        transition_list.append([node,node,float(probability)])
    
    transition_list.sort()
    
    return transition_list

def calculate_transition_matrix(energy,beta):
    
    N = len(energy)
    
    transition_matrix = np.zeros([N,N])
    
    n = int(np.log2(N))
    
    for node in range(N):
        
        node_probability = 0
        
        for spin in range(n):
            
            neighbor = list(format(node, f'0{n}b'))
            
            neighbor[spin] = str(int(neighbor[spin])^1)
            
            neighbor = ''.join(neighbor)
            
            neighbor = int(neighbor,2)
            
            energy_diff = energy[node] - energy[neighbor]
            
            probability = min(1,np.exp(beta*energy_diff)) / n
            
            transition_matrix[neighbor,node] = probability
            
            node_probability += probability
        
        probability = 1 - node_probability
        
        if probability < 0: probability = 0
        
        transition_matrix[node,node] = probability
    
    return transition_matrix


def calculate_exact_distribution(energy,beta):
    
    exp = np.exp(-beta*energy)
    
    Z = sum(exp)
    
    probs = exp/Z
    
    return probs