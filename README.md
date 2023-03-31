# LPC-based cross synthesis

The code implements a vocoder through cross synthesis.
The coefficients for the filters are computed in two different ways:
- Solving Wiener-Hopf equations
- Using Steepest Descent algorithm

## Getting Started

To use this code, simply run the script `lpc.py`.

In the `res` folder should be located `piano.wav` and `speech.wav`

## Functions and implementation

The function `perform_lpc()` is the main function of this script.

For Overlap and Add we chose `Hanning` window and `overlap_factor = 0.5`

Changing the global parameters is possible to change the behaviour of the script.
The given values are the one we found as optimal for quality and speed of the algorithms.

No high-level functions have been used.
The signal is divided in frames and windowed at the beginning of the process.
Each frame is analyzed individually. 
Through the chosen algorithm, LPC coefficients are computed and with that, the relative whitening filters coefficients.

The convolution is done through moltiplicatin in frequency domain.

The zero padding is not calculated through the formula but overdone. Zero-padding to obtain the length of `frame_length`+`filter1_legth`+`filter2_length` wasn't sufficient to avoid artifacts.

To obtain the shaper filter from the voice we computed `1/whitening` in frequency domain.

The inverse ffts are then summed back and written as wav file with the name of `output.wav`

Before all the process, the data is normalized to avoid overflow errors in the Steepest Descent algorithm

## Group
- Guglielmo Fratticioli
- Elia Pirrello