'''
Code which implements the Bandt-Pompe algorithm to obtain the ordinal pattern 
probability distribution from the time series, with the obtained probability 
distribution the statistical complexity measures Permutation Entropy and 
Jensen-Shannon Complexity measure are calculated for use in the 
Complexity-Entropy analysis. 

'''


import numpy as np
import math
from tqdm import tqdm


class ComplexityEntropy():
    ''' Class containing appropriate functions to calculate the complexity and entopy values for a given time sereis '''

    def __init__(self, time_series, d, tau = 1):
        '''
        Parameters
        ----------
        time_series : LIST / ARRAY
            Time series
        d : INT
            Embedding dimension
        tau : INT, optional
            Embedding delay. The default is 1.

        Returns
        -------
        None. Initializing function

        '''
        self.time_series = np.array(time_series)
        self.d = d
        self.tau = tau


    def Permutation_frequency(self):
        '''
        Function that calculates the relative frequnecy for the different permutations
        of the time series for the given embedding dimension

        Returns
        -------
        relative_frequency/relative_frequency_pad : ARRAY
            Probability distribution for the possible ordinal patterns, Padded with
            zeros to have length d!

        '''
        Possible_permutaitons = math.factorial(self.d) #list(itertools.permutations(range(self.d)))
        perm_list = []

        # Implementing embedding delay on the time series
        self.time_series = self.time_series[::self.tau]

        for i in range(len(self.time_series) - self.d + 1):
            # "permutation of emb_dimension sized segments of the time series"
            permutation = list(np.argsort(self.time_series[i : (self.d + i)]))
            perm_list.append(permutation)

        # Find the different permutations and calculates their number of appearance
        elements, frequency = np.unique(np.array(perm_list), return_counts = True, axis = 0)

        # Divides by the total number of permutations, gets relative frequency / "probalbility" of appearance
        relative_frequency = np.divide(frequency, (len(self.time_series) - self.tau * (self.d - 1)))

        # If the two arrays do not have the same shape, add zero padding to make their lengths equal
        if len(relative_frequency) != Possible_permutaitons:
            relative_frequency_pad = np.pad(relative_frequency, (0, int(Possible_permutaitons - len(relative_frequency))), mode = 'constant')
            return relative_frequency_pad

        else:
            return relative_frequency

    def Permutation_Entropy(self, Permutation_probability):
        '''
        Function to calculate the permutation entropy for a given probability distribution
        Retruns the normalized Shannon entropy

        Parameters
        ----------
        Permutation_probability : ARRAY
            Array contaning the probalbility distribution of the ordinal patterns.

        Returns
        -------
        permutation_entropy : FLOAT
            Entropy value of the time series.

        '''
        permutation_entropy = 0.0

        # Calculate the max entropy, max = log(d!)
        max_entropy = np.log2(len(Permutation_probability))

        for p in Permutation_probability:
            if p != 0.0:
                permutation_entropy += p * np.log2(p)
        return - permutation_entropy/max_entropy

    def Shannon_Entropy(self, Permutation_probability):
        '''
        Regular Shannon entropy, not normalized

        Parameters
        ----------
        Permutation_probability : ARRAY
            Array contaning the probalbility distribution of the ordinal patterns.

        Returns
        -------
        shannon_entropy : FLOAT
            Shannon entropy value.

        '''
        shannon_entropy = 0.0
        for p in Permutation_probability:
            if p != 0.0:
                shannon_entropy += p * np.log2(p)
        return -shannon_entropy

    def Jensen_Shannon_Complexity(self, Permutation_probability):
        '''
        Function to calculate the Jensen-Shannon complexity value for the time sereis

        Parameters
        ----------
        Permutation_probability : ARRAY
            Array contaning the probalbility distribution of the ordinal patterns.

        Returns
        -------
        jensen_shannon_complexity : FLOAT
            Jensen-Shannon complexity value.

        '''
        P = Permutation_probability
        N = len(P)
        C1 = (N + 1)/N * np.log2(N + 1)
        C2 = 2 * np.log2(2*N)
        C3 = np.log2(N)
        PE = self.Permutation_Entropy(P)

        P_uniform = []
        for i in range(N):
            P_uniform.append(1/N)

        JS_div = self.Shannon_Entropy((P + P_uniform)*0.5) - 0.5 * self.Shannon_Entropy(P) - 0.5 * self.Shannon_Entropy(P_uniform)
        jensen_shannon_complexity = -2 * (1/(C1 - C2 + C3)) * JS_div * PE
        return jensen_shannon_complexity

    def CH_plane(self):
        '''
        Computes the permutation entropy and the Jensen-Shannon complexity for the time series
        with the functions defined in the class

        Returns
        -------
        permutation_entropy : FLOAT
            Permutation entropy value of the time series.
        jensen_shannon_complexity : FLOAT
            Jensen-Shannon complexity value of the time series.

        '''
        # Calling the function to generate the relative frequency for the ordinal patterns
        relative_frequency = self.Permutation_frequency()

        #Using relative frequency to calculate the entropy/complexity for the time series
        permutation_entropy = self.Permutation_Entropy(relative_frequency)
        jensen_shannon_complexity = self.Jensen_Shannon_Complexity(relative_frequency)

        return permutation_entropy, jensen_shannon_complexity


class MaxMin_complexity(ComplexityEntropy):
    '''
    Class containing the fuctions to calculate the maximum complexity and
    minimum complexity lines for the Complexity-Entropy plane with
    embedding dimension d
    '''

    def __init__(self, d, n_steps = 500):
        '''

        Parameters
        ----------
        d : INT
            Embedding dimension.

        Returns
        -------
        None.

        '''
        # definin class variables available to all functions/methods contained in this class
        self.d = d
        self.N = math.factorial(self.d)
        self.n_steps = n_steps
        self.d_step = (1 - 1/self.N) / (self.n_steps)

        # Initilalizing __init__()-function to parent (ComplexityEntropy class)
        # class to make the functions contained in that class available to this class
        super().__init__(time_series = None, d = self.d)

        # Lists to contain the x and y values for the minimum and maximum
        # complexity lines
        self.min_complexity_entropy_x = list()
        self.min_complexity_entropy_y = list()
        self.max_complexity_entropy_x = list()
        self.max_complexity_entropy_y = list()


    def Minimum(self):
        '''
        Function to calculate the minimum complexity line

        Returns
        -------
        min_complexity_entropy_x : LIST
            x-values for the minimum complexity line.

        min_complexity_entropy_y : LIST
            y-values for the minimum complexity line.

        '''
        p_min = list(np.arange(1/self.N, 1, self.d_step))
        for n in tqdm(range(len(p_min)), desc='Minimum', ncols=70):
            P_minimize = []
            if p_min[n] > 1:
                p_min[n] = 1
            P_minimize.append(p_min[n])
            for i in range(self.N - 1):
                p_rest = (1 - p_min[n]) / (self.N - 1)
                P_minimize.append(p_rest)

            # Convert from list structure to array structure
            P_minimize = np.array(P_minimize)

            # Adding the calculated x and y (entropy and complexity) values
            # to their approppriate lists
            self.min_complexity_entropy_x.append(self.Permutation_Entropy(P_minimize))
            self.min_complexity_entropy_y.append(self.Jensen_Shannon_Complexity(P_minimize))

        return self.min_complexity_entropy_x, self.min_complexity_entropy_y

    def Maximum(self):
        '''
        Function to calculate the maximum complexity line

        Returns
        -------
        max_complexity_entropy_x : FLOAT
            x-values for the maximum complexity line.

        max_complexity_entropy_y : FLOAT
            y-values for the maximum complexity line.

        '''

        for n in tqdm(range(self.N - 1), desc='Maximum', ncols=80):
            p_max = list(np.arange(0, 1 / (self.N - n), self.d_step))

            for m in range(len(p_max)):
                P_maximize = list()
                P_maximize.append(p_max[m])
                p_rest = (1 - p_max[m]) / (self.N - n - 1)
                for i in range(self.N - n - 1):
                    P_maximize.append(p_rest)

                if len(P_maximize) != self.N:
                    P_maximize = np.pad(P_maximize, (0, n), mode = 'constant')

                # Convert from list structure to array structure
                P_maximize = np.array(P_maximize)

                #Adding the calculated x and y (entropy and complexity) values
                # to their approppriate lists
                self.max_complexity_entropy_x.append(self.Permutation_Entropy(P_maximize))
                self.max_complexity_entropy_y.append(self.Jensen_Shannon_Complexity(P_maximize))

        return self.max_complexity_entropy_x, self.max_complexity_entropy_y
