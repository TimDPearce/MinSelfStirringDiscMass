# MinSelfStirringDiscMass

Python program to calculate the absolute minimum mass a debris disc 
requires to self-stir, by Tim D. Pearce. The model is fully described in
Pearce et al. 2022 (Sect. 4.1), and essentially combines the findings of
Krivov & Booth 2018 with those of Krivov & Wyatt 2021. This yields a 
self-consistent model for self-stirring, which is used here to find the
minimum possible mass that a debris disc requires to self-stir. Given the
parameters of the star, the disc edge locations, and the mass in 
millimeter dust (and optionally their associated uncertainties), the 
program calculates the minimum possible self-stirring mass, and the 
minimum SMax (radius of the largest disc body) required to self-stir. For 
the minimum-mass, self-stirring disc, the program also outputs the radius 
of the largest colliding particle (SKm), and the collision speed required 
to fragment that particle, vFrag(SKm).

To use the program, simply change the values in the 'User Inputs' section
just below. You should not have to change anything outside of that 
section. The default settings are for HD38206.

Feel free to use this code, and if the results go into a publication,
then please cite Pearce et al. 2022. Finally, let me know if you find any
bugs or have any requests!
