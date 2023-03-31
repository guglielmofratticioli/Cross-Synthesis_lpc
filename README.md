# weiner_lpc
linear predictive coding according to Weiner theory  

symmetric windows -> for filter design (avoid for real time). scipy. and periodic window. how to get -> reciprocal of FFT of h_n

# LPC-based cross synthesis

The code implements a vocoder through cross synthesis.
The coefficients for the filters are computed in two different ways:
- Solving Wiener-Hopf equations
- 

## Getting Started

To use this code, simply clone the repository and run the `perform_lpc` function in the `__main__` block of `main.py`. 

```python
if __name__ == '__main__' : 
   perform_lpc()