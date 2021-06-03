# Complexity-Entropy Analysis
The code given here implements the Bandt-Pompe algorithm to obtain a probability distribution based on the amplitude ordinal pattern of a given time series. Once the probability distribution is calculated, Permutation Entropy and Jensen-Shannon statistical complexity measure can be obtained.

Interpreting the entropy and complexity measures as coordinate axis, the measures define a representation space, the so-called Complexity-Entropy plane (often denoted as CH-plane). The Complexity-Entropy is then used as a graphical time series analysis tool which has the ability to distinguish between time series gathered from stochastic systems, and time series gathered from chaotic systems. In addition, the Complexity-Entropy plane has the ability to separate time series with trends and time series of noise.  

The code is implemented object-oriented and contains a class containing functions to calculate the permutation entropy and Jensen-Shannon complexity. The other class calculates the maximum and minimum complexity lines using synthetically made probability distributions.

For more information and theory on the entropy and complexity measures, and the theory behind the probability distributions which maximizes and minimizes the complexity measures, see the pdf *Theory*.
## Dependencies
Install requirements with
```sh
pip install -r requirements.txt
```
## Use
The class *ComplexityEntropy* is initialized with the time series, and the embedding dimension used for the Complexity-Entropy analysis as arguments. An additional argument, embedding delay, can also be passed as an argument, default value is set to one meaning no delay.

If you have a time series, named *time_series*, one can initialize the class and calculate the permutation entropy (*H*) and Jensen-Shannon complexity (*C*) with the following code, For this example, the object is named *CH_TimeSeries*:

```sh
#Embedding dimension
d = 6

# Creating class object, calulating the entropy and complexity measure
CH_TimeSeries = ComplexityEntropy(time_series, d)
H, C = CH_TimeSeries.CH_plane()
```

To obtain the maximum and minimum complexity lines, use the following example code:
```sh
d = 6

CH_region = MaxMin_complexity(d, n_steps = 1)
Max = CH_region.Maximum()
CH_region = MaxMin_complexity(d, n_steps = 100)
Min = CH_region.Minimum()
```
The *n_steps* arguments given above works for embedding dimension 6 and above. For embedding dimension below 6 the values need to be tweaked to obtain reasonable line shapes.

The Complexity-Entropy plane can be plotted with the maximum and minimum complexity line with the complexity and entropy measure of *time_series* marked on the plane.  
```sh
plt.plot(Max[0], Max[1], color = 'tab:blue')
plt.plot(Min[0], Min[1], color = 'tab:blue')
plt.plot(H, C)
plt.show()
```
