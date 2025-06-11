# Code-Nanophysics
Some written, by me and my lab mate, codes for data analysys of Gold (semi)spherical nanoparticle, for the course Introduction to Nanophyisics held by Professor G.Mattei in Padua in 2024-2025


This repository contains a Python-based fitting routine designed to analyze the optical absorbance spectra of metal nanoparticles suspended in solution. The code models experimental absorbance data using the Lambert-Beer law in conjunction with Gans theory, which accurately describes the optical response of ellipsoidal nanoparticles by accounting for their anisotropic shape.

 What the code does
Fits experimental absorbance spectra of gold nanoparticles (Au NPs) using the Lambert-Beer law.
Computes the extinction cross-section from first principles using:
Size-corrected dielectric functions
Gans theory for ellipsoidal particles

Allows automated optimization of:
-Particle radius (major axis a)
-Eccentricity (via fixed ratio a/b)
-Particle concentration
-Medium dielectric constant

As input:
-A .txt file containing the measured absorbance spectrum (wavelength in nm, absorbance)
-Optical constants of the nanoparticle material (e.g., Johnson-Christy data for gold)
