
'''Python program to calculate the absolute minimum mass a debris disc 
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
bugs or have any requests!'''

############################### Libraries ###############################
import numpy as np
import math
from scipy.optimize import minimize
############################## User Inputs ##############################
'''Parameters of the system, and their associated uncertainties and 
units. For example, if the star mass is 1.2 MSun with 1sigma 
uncertainties of +0.1 MSun and -0.2 MSun, then set mStar_mSun=2, 
mStar1SigUp_mSun=0.1 and mStar1SigDown_mSun=0.2. If you do not want
to consider an uncertainty, use numpy NaN, e.g. 
mStar1SigUp_mSun = np.nan. You should not need to change any part of 
the code other than that in this section.'''

# Star mass, in Solar masses
mStar_mSun = 2.36
mStar1SigUp_mSun = 0.02
mStar1SigDown_mSun = 0.02

# System age, in Myr
age_Myr = 42.
age1SigUp_Myr = 6.
age1SigDown_Myr = 4.

# Apocentre of the disc INNER edge, in au (if disc is axisymmetric, then 
# this is just the inner edge radius)
discInnerEdgeApo_au = 140.
discInnerEdgeApo1SigUp_au = 30.
discInnerEdgeApo1SigDown_au = 40.

# Pericentre of the disc OUTER edge, in au (if disc is axisymmetric, then 
# this is just the outer edge radius)
discOuterEdgePeri_au = 190.
discOuterEdgePeri1SigUp_au = 30.
discOuterEdgePeri1SigDown_au = 30.

# Dust mass in millimeter grains, in Earth masses (#NOTE: program cannot
# currently take asymmetric dust mass uncertainties. This will be allowed 
# soon!)
mDust_mEarth = 0.07
mDustError_mEarth = 0.01

################ Global variables (change at your peril!) ###############
# Size distribution indicies for the 3-power-law model of 
# Krivov & Wyatt 2021
q = 3.5
qMed = 3.7
qBig = 2.8
qBigError = 0.1

############################ Maths functions ############################
def RoundNumberToDesiredSigFigs(num, sigFigs=None):
	'''Returns a float of a given number to the specified significant 
	figures'''

	# Default number of decimal digits (in case precision unspecified)
	defaultSigFigs = 2

	# Get sig figs if not defined
	if sigFigs is None:
		sigFigs = defaultSigFigs
			
	# Catch case if number is zero
	if num == 0:
		exponent = 0
	
	# Otherwise number is non-zero
	else:
		exponent = GetBase10OrderOfNumber(num)
	
	# Get the coefficient
	coefficient = round(num / float(10**exponent), sigFigs-1)

	roundedNumber = coefficient * 10**exponent
	
	# Get the decimal places to round to (to avoid computer rounding errors)
	decimalPlacesToRoundTo = sigFigs-exponent
	
	roundedNumber = round(roundedNumber, decimalPlacesToRoundTo)
	
	return roundedNumber
	
#------------------------------------------------------------------------
def GetBase10OrderOfNumber(number):
	'''Return the order of the positive number in base 10, e.g. inputting
	73 returns 1 (since 10^1 < 73 < 10^2).'''
	
	if number <= 0: return np.nan
	
	return int(math.floor(np.log10(abs(number))))

#------------------------------------------------------------------------
def GetGaussianAsymmetricUncertainties(dependencyUncertainties1SigUp_unit, dependencyUncertainties1SigDown_unit, differentialValueWRTDependencies_unitPerUnit):
	'''Get the ~Gaussian uncertainties on a value, where its 
	dependencies may have asymmetric errors. Takes dictionaries of the
	positive and negative untertainties on the dependencies, and the
	derivatives of the value wrt its dependencies'''
	
	# Contributions to the total squared errors from the different 
	# sources
	value1SigUpSquared_unit2, value1SigDownSquared_unit2 = 0., 0.

	for dependencyName in dependencyUncertainties1SigUp_unit:
		dependencyValue1SigUp_unit = dependencyUncertainties1SigUp_unit[dependencyName]
		dependencyValue1SigDown_unit = dependencyUncertainties1SigDown_unit[dependencyName]
		
		differentialValueWRTDependency_unitPerUnit = differentialValueWRTDependencies_unitPerUnit[dependencyName]
	
		# Catch overflow error if derivatives very large (e.g. 
		# Laplace coefficient as alpha->1)
		differentialValueWRTDependency_unitPerUnit = min(differentialValueWRTDependency_unitPerUnit, 1e99)
		differentialValueWRTDependency_unitPerUnit = max(differentialValueWRTDependency_unitPerUnit, -1e99)
			
		# If the value increases as this dependency increases, then 
		# positive value error corresponds to positive dependency error etc.
		if differentialValueWRTDependency_unitPerUnit >= 0:	
			value1SigUpSquared_unit2 += (differentialValueWRTDependency_unitPerUnit * dependencyValue1SigUp_unit)**2
			value1SigDownSquared_unit2 += (differentialValueWRTDependency_unitPerUnit * dependencyValue1SigDown_unit)**2
		
		# Otherwise the plt mass decreases as this variable 
		# increases, so positive mass error corresponds to negative 
		# variable error etc.
		else:
			value1SigUpSquared_unit2 += (differentialValueWRTDependency_unitPerUnit * dependencyValue1SigDown_unit)**2
			value1SigDownSquared_unit2 += (differentialValueWRTDependency_unitPerUnit * dependencyValue1SigUp_unit)**2

	# Finally, get the plt mass errors
	value1SigUp_unit = value1SigUpSquared_unit2**0.5
	value1SigDown_unit = value1SigDownSquared_unit2**0.5

	return value1SigUp_unit, value1SigDown_unit
		
########################### Dynamics functions ##########################
def GetDiscParsAndErrorsForMinDiscMassToSelfStir_mEarthAndKm(mDust_mEarth, mDustError_mEarth, age_Myr, age1SigUp_Myr, age1SigDown_Myr, mStar_mSun, mStar1SigUp_mSun, mStar1SigDown_mSun, discInnerEdge_au, discInnerEdge1SigUp_au, discInnerEdge1SigDown_au, discOuterEdge_au, discOuterEdge1SigUp_au, discOuterEdge1SigDown_au, sMm_mm=1., debug=False):
	'''Get the parameters of the minimum disc mass required for the disc
	to self-stir, from Krivov & Wyatt 2021 and Krivov & Booth 2018. 
	Changing SMax changes both the disc mass and the disc mass required 
	to self-stir, and function uses numerical optimisation to identify 
	the minimum SMax for which the disc mass is at least that required to
	 self-stir.'''
	
	# Get the minimum SMax required to stir the disc
	minSMaxToStirDisc_km = GetMinSMaxToStirDisc_km(mDust_mEarth, age_Myr, mStar_mSun, discInnerEdge_au, discOuterEdge_au)

	# Get the disc mass corresponding to this SMax. Done this way, 
	# rather than just using disc mass at this SMax, so consistency can 
	# be checked later
	discRadius_au = 0.5*(discInnerEdge_au+discOuterEdge_au)
	
	mDiscSkm_mEarth, sKmToStir_km, mDisc1km_mEarth = GetDiscMassAndSkmFromSMax_mEarthAndKm(minSMaxToStirDisc_km, mDust_mEarth, age_Myr, mStar_mSun, discRadius_au)
	
	minDiscMassToSelfStir_mEarth, vFragSkm_kmPerS = GetDiscMassToSelfStirAndVFragFromSMaxAndSkm_mEarthAndKmPerS(minSMaxToStirDisc_km, sKmToStir_km, mStar_mSun,
		age_Myr, discInnerEdge_au, discOuterEdge_au)
	
	# Package the results
	discParsForMinDiscMassToSelfStir = {'minDiscMassToSelfStir_mEarth': minDiscMassToSelfStir_mEarth,
								'minSMaxToStirDisc_km': minSMaxToStirDisc_km,
								'sKmToStir_km': sKmToStir_km,
								'vFragSkm_kmPerS': vFragSkm_kmPerS}

	# Debug code
	if debug:
		print(discParsForMinDiscMassToSelfStir)
									
	# Get the uncertainties on all parameters
	uncertaintiesOnDiscParsForMinDiscMassToSelfStir = GetUncertaintiesOnDiscParsForMinDiscMassToSelfStir(discParsForMinDiscMassToSelfStir, mDust_mEarth, mDustError_mEarth, age_Myr, age1SigUp_Myr, age1SigDown_Myr, mStar_mSun, mStar1SigUp_mSun, mStar1SigDown_mSun, discInnerEdge_au, discInnerEdge1SigUp_au, discInnerEdge1SigDown_au, discOuterEdge_au, discOuterEdge1SigUp_au, discOuterEdge1SigDown_au, sMm_mm, mDisc1km_mEarth, debug=debug)
	
	# Combine the value and uncertainty dictionaries
	discParsAndErrorsForMinDiscMassToSelfStir = {**discParsForMinDiscMassToSelfStir, **uncertaintiesOnDiscParsForMinDiscMassToSelfStir}

	# Error check
	if sKmToStir_km > minSMaxToStirDisc_km:
		print('### WARNING: sKm_km (%s) > sMax_km (%S)!' % (sKmToStir_km, minSMaxToStirDisc_km))
		
	return discParsAndErrorsForMinDiscMassToSelfStir
	
#------------------------------------------------------------------------
def GetMinSMaxToStirDisc_km(mDust_mEarth, age_Myr, mStar_mSun, discInnerEdge_au, discOuterEdge_au, sMm_mm=1.):
	'''Get the minimum SMax required for the disc to self-stir, from 
	Krivov & Wyatt 2021 and Krivov & Booth 2018. Changing SMax changes 
	both the disc mass and the disc	mass required to self-stir, and 
	function uses numerical optimisation to identify the minimum SMax for
	which the disc mass is at least that required to self-stir.'''
	
	# Initial guess at SMax
	sMaxInitial_km = 200.
	
	# Work in log space, so numerical solver more stable
	log10SMaxInitial_km = np.log10(sMaxInitial_km)
	
	# Find the SMax for which the disc mass equals the mass required to 
	# self-stir
	minimisationFunction = minimize(GetAbsDifferenceBetweenDiscMassAndStirringMass_mEarth, log10SMaxInitial_km, args = (mDust_mEarth, sMm_mm, age_Myr, mStar_mSun, discInnerEdge_au, discOuterEdge_au), bounds=[(-3,6)])
	
	log10SMax_km = minimisationFunction.x[0]
	sMax_km = 10**log10SMax_km

	return sMax_km
	
#------------------------------------------------------------------------
def GetAbsDifferenceBetweenDiscMassAndStirringMass_mEarth(log10SMax_km, mDust_mEarth, sMm_mm,
		age_Myr, mStar_mSun, discInnerEdge_au, discOuterEdge_au):
	'''For the given SMax, get the difference between the disc mass and 
	that required for self-stirring. Used in the numerical evaluation of 
	the minimum Smax for stirring'''

	# Work in log space, so numerical evaluation more stable
	sMax_km = 10**log10SMax_km
	
	discRadius_au = 0.5*(discInnerEdge_au + discOuterEdge_au)
	
	# Get the disc mass	and Skm from SMax
	mDisc_mEarth, sKm_km, mDisc1km_mEarth = GetDiscMassAndSkmFromSMax_mEarthAndKm(sMax_km, mDust_mEarth, age_Myr, mStar_mSun, discRadius_au)
	
	# Get the disc mass required to self-stir from SMax and Skm
	mDiscStir_mEarth = GetDiscMassToSelfStirAndVFragFromSMaxAndSkm_mEarthAndKmPerS(sMax_km, sKm_km, mStar_mSun,
		age_Myr, discInnerEdge_au, discOuterEdge_au)[0]
	
	# Get the difference between the disc mass and that required to 
	# self-stir
	massDifference_mEarth = abs(mDiscStir_mEarth - mDisc_mEarth)
	
	return massDifference_mEarth
	
#------------------------------------------------------------------------
def GetDiscMassAndSkmFromSMax_mEarthAndKm(sMax_km, mDust_mEarth, age_Myr, mStar_mSun, discRadius_au, sMm_mm = 1.):
	'''Get the disc mass and Skm from SMax, according to the 3-power-law
	 size distribution from Krivov & Wyatt 2021 (eq 9), where the 
	 time-dependent Skm has also been found (Krivov & Wyatt 2021, 
	 eqs 9, 17, 19).'''
	
	# Get the disc mass
	mDisc1km_mEarth = Get3PowerLawDiscMass_mEarth(mDust_mEarth, sMm_mm, 1., sMax_km)
	mDiscSkm_mEarth = GetMDiscCorrectedForSkm_mEarth(age_Myr, mDisc1km_mEarth, mStar_mSun, discRadius_au)

	# Get sKm
	sKm_km = GetSkm_km(mDisc1km_mEarth, mDiscSkm_mEarth)
		
	return mDiscSkm_mEarth, sKm_km, mDisc1km_mEarth	
	
#------------------------------------------------------------------------
def Get3PowerLawDiscMass_mEarth(mDust_mEarth, sMm_mm, sKm_km, sMax_km):
	'''Get the disc mass from the dust mass, according to the 
	3-power-law model (Krivov & Wyatt 2021, eq 9)'''
	
	# Get the non-sKm and -sMax terms in the equation
	mDiscDividedBySKmAndSMaxTerms_mEarth = GetDiscMassDividedBySKmAndSMaxTerms_mEarth(mDust_mEarth, sMm_mm)
	
	# Multiply by the sKm and sMax terms
	mDisc_mEarth = mDiscDividedBySKmAndSMaxTerms_mEarth * sKm_km**(qBig - qMed) * sMax_km**(4.-qBig)

	return mDisc_mEarth	
	
#------------------------------------------------------------------------
def GetDiscMassDividedBySKmAndSMaxTerms_mEarth(mDust_mEarth, sMm_mm):
	'''Get the non-sKm and -sMax terms in the 3-power-law disc mass 
	equation (Krivov & Wyatt 2021, eq 9)'''
	
	mDiscDividedBySKmAndSMaxTerms_mEarth = mDust_mEarth * 10**(6.*(4.-qMed)) * (4.-q)/(4.-qBig) * sMm_mm**(qMed-4.)
	
	return mDiscDividedBySKmAndSMaxTerms_mEarth
	
#------------------------------------------------------------------------
def GetMDiscCorrectedForSkm_mEarth(age_Myr, mDisc1km_mEarth, mStar_mSun, discRadius_au):
	'''Calculate the disc mass corrected for the time-dependent 
	transition size sKm (Krivov & Wyatt 2021, eq 19). This is denoted MD*
	in Krivov & Wyatt 2021.'''

	# Get the non-mDisc1km terms in the equation
	discMassCorrectedForSKmDividedByDiscMass1KmTerm_mEarth = GetDiscMassCorrectedForSKmDividedByDiscMass1KmTerm_mEarth(age_Myr, mStar_mSun, discRadius_au)

	# Multiply by the mDisc1km_mEarth term
	mDiscAst_mEarth = discMassCorrectedForSKmDividedByDiscMass1KmTerm_mEarth * mDisc1km_mEarth**.41

	return mDiscAst_mEarth
	
#------------------------------------------------------------------------
def GetDiscMassCorrectedForSKmDividedByDiscMass1KmTerm_mEarth(age_Myr, mStar_mSun, discRadius_au):
	'''Get the terms in the disc mass corrected for the time-dependent 
	transition size sKm (Krivov & Wyatt 2021 eq 19, there denoted MD*) 
	that do not depend on the disc mass derived for sKm = 1km'''
	
	# Convert to units in equation
	age_Gyr = age_Myr / 1000.
	
	discMassCorrectedForSKmDividedByDiscMass1KmTerm_mEarth = 2.66372e-4 * age_Gyr**-.59 * mStar_mSun**-.79 * discRadius_au**2.56

	return discMassCorrectedForSKmDividedByDiscMass1KmTerm_mEarth	
	
#------------------------------------------------------------------------
def GetSkm_km(mDisc1km_mEarth, mDiscSkm_mEarth):
	'''Get the transition size sKm, which is time-dependent 
	(Krivov & Wyatt 2021, eq 17)'''

	sKm_km = (mDiscSkm_mEarth / mDisc1km_mEarth)**(1./(qBig-qMed))

	return sKm_km
	
#------------------------------------------------------------------------
def GetDiscMassToSelfStirAndVFragFromSMaxAndSkm_mEarthAndKmPerS(sMax_km, sKm_km, mStar_mSun,
		age_Myr, discInnerEdge_au, discOuterEdge_au):
	'''Get the disc mass reqired to self-stir, for given values of sMax 
	and sKm. Method calculates the fragmentation speed for the weakest 
	body currently in the collisional cascade; if the largest colliding
	body skm is in the strength regime, then this is vFrag(sKm). For 
	larger sKm, the weakest body will be at the strength -> gravity 
	transition at s~0.1km, in which case take vFrag(s~0.1km). Then use
	Krivov & Booth 2018 (eq 28) to find min mass to stir.'''
	
	# Get the size of the weakest body in the cascade
	sTransition_km = 0.110642
	sWeakest_km = min(sTransition_km, sKm_km)
	
	# Get the fragmentation velocity of the weakest body
	vFragWeakest_kmPerS = GetFragmentationVelocity_kmPerS(sWeakest_km)
	vFragWeakest_mPerS = vFragWeakest_kmPerS*1000
		
	# Get the disc mass required to self-stir for this SMax
	mDiscStir_mEarth = GetKrivovBoothDiscMassForSelfStirring_mEarth(mStar_mSun, np.nan, 
		age_Myr, np.nan, discInnerEdge_au, np.nan, np.nan, discOuterEdge_au, np.nan, np.nan,
		fragmentationSpeed_mPerS = vFragWeakest_mPerS, maxColliderRadius_km = sMax_km)[0]	
	
	return mDiscStir_mEarth, vFragWeakest_kmPerS	
	
#------------------------------------------------------------------------
def GetFragmentationVelocity_kmPerS(bodyRadius_km):
	'''Calculate the fragmentation velocity for a body of a given 
	radius, assuming the given QD* prescription'''
	
	# Get QD* divided by the impact velocity**.5
	qDStarDividedBySquareRootVelocity_JS05PerM05Kg = GetQDStarDividedBySquareRootVelocity_JS05PerM05Kg(bodyRadius_km)

	# Get the fragmentation velocity
	vFrag_mPerS = (8.*qDStarDividedBySquareRootVelocity_JS05PerM05Kg)**(2./3.)

	# Convert to km/s
	vFrag_kmPerS = vFrag_mPerS / 1000.
	
	return vFrag_kmPerS	
	
#------------------------------------------------------------------------
def GetQDStarDividedBySquareRootVelocity_JS05PerM05Kg(bodyRadius_km):
	'''Get QD*/vimp**.5, i.e. the critical fragmention energy
	(Krivov+18 eq 1) divided by the square root of the impact velocity.
	Assumes basalt from SPH simulations by Benz & Asphaug (1999), plus 
	the velocity dependence from Stewart & Leinhardt (2009).'''

	# Coefficients in equation
	As_JPerKg = 500.
	Ag_JPerKg = As_JPerKg
	bs = -.12
	bg = .46
	v0_mPerS = 3000.
		
	# Get QD* divided by the impact velocity**.5
	qDStarDividedBySquareRootVelocity_JS05PerM05Kg = (As_JPerKg*(bodyRadius_km*1000.)**(3*bs) + Ag_JPerKg*(bodyRadius_km)**(3*bg)) / v0_mPerS**.5

	return qDStarDividedBySquareRootVelocity_JS05PerM05Kg
	
#------------------------------------------------------------------------
def GetKrivovBoothDiscMassForSelfStirring_mEarth(starMass_mSun, starMassError_mSun, starAge_Myr, starAgeError_Myr, discInnerEdge_au, discInnerEdge1SigUp_au, discInnerEdge1SigDown_au, discOuterEdge_au, discOuterEdge1SigUp_au, discOuterEdge1SigDown_au, gamma = 1.5, solidDensity_gPerCm3 = 1.0, fragmentationSpeed_mPerS = 30., maxColliderRadius_km = 200.):
	'''Returns the minimum disc mass required to self-stir a debris disc 
	within the system age, according to Krivov & Booth (2018). Their 
	equation 28 is re-written in terms of disc edges, and re-arranged 
	for disc mass. Default values are those from Krivov & Booth (2018), 
	with gamma=1.5 as in MussoBarcucci2021.'''	

	# Combinations of the inner and outer edges	
	innerEdgePlusOuterEdge_au = discInnerEdge_au + discOuterEdge_au
	outerEdgeMinusInnerEdge_au = discOuterEdge_au - discInnerEdge_au 

	# If the disc has zero width, then this stirring calculation cannot 
	# be done
	if outerEdgeMinusInnerEdge_au <= 0:
		return np.nan, np.nan, np.nan
	
	# Get the minimum disc mass
	minDiscMass_mEarth = 1.644023e-4 * starAge_Myr**-1 * gamma**-1 * (solidDensity_gPerCm3/1.0)**-1 * (fragmentationSpeed_mPerS / 30.)**4 * (maxColliderRadius_km / 200.)**-3 * starMass_mSun**-.5 * outerEdgeMinusInnerEdge_au * innerEdgePlusOuterEdge_au**(5./2.)

	# Get the errors on the minimum disc mass		
	starMassErrorCont = -1./2. * starMassError_mSun/starMass_mSun
	starAgeErrorCont = -1. * starAgeError_Myr/starAge_Myr	
	
	discOuterEdgeErrorCoefficient = 5./2. * discOuterEdge_au / innerEdgePlusOuterEdge_au + discOuterEdge_au / outerEdgeMinusInnerEdge_au
	discInnerEdgeErrorCoefficient = 5./2. * discInnerEdge_au / innerEdgePlusOuterEdge_au - discInnerEdge_au / outerEdgeMinusInnerEdge_au

	outerEdge1SigUpErrCont = discOuterEdgeErrorCoefficient*discOuterEdge1SigUp_au/discOuterEdge_au
	outerEdge1SigDownErrCont = discOuterEdgeErrorCoefficient*discOuterEdge1SigDown_au/discOuterEdge_au
			
	# Need to know whether disc mass increases or decreases with disc 
	# inner edge
	if discInnerEdgeErrorCoefficient >= 0:
		innerEdge1SigUpErrCont = discInnerEdgeErrorCoefficient*discInnerEdge1SigUp_au/discInnerEdge_au
		innerEdge1SigDownErrCont = discInnerEdgeErrorCoefficient*discInnerEdge1SigDown_au/discInnerEdge_au

	else:
		innerEdge1SigUpErrCont = discInnerEdgeErrorCoefficient*discInnerEdge1SigDown_au/discInnerEdge_au
		innerEdge1SigDownErrCont = discInnerEdgeErrorCoefficient*discInnerEdge1SigUp_au/discInnerEdge_au
		
	minDiscMass1SigUp_mEarth = minDiscMass_mEarth * (starMassErrorCont**2 + starAgeErrorCont**2 + outerEdge1SigUpErrCont**2 + innerEdge1SigUpErrCont**2)**0.5
	minDiscMass1SigDown_mEarth = minDiscMass_mEarth * (starMassErrorCont**2 + starAgeErrorCont**2 + outerEdge1SigDownErrCont**2 + innerEdge1SigDownErrCont**2)**0.5 
	
	return minDiscMass_mEarth, minDiscMass1SigUp_mEarth, minDiscMass1SigDown_mEarth

#------------------------------------------------------------------------
def GetUncertaintiesOnDiscParsForMinDiscMassToSelfStir(discParsForMinDiscMassToSelfStir, mDust_mEarth, mDustError_mEarth, starAge_Myr, starAge1SigUp_Myr, starAge1SigDown_Myr, starMass_MSun, starMass1SigUp_MSun, starMass1SigDown_MSun, discInnerEdge_au, discInnerEdge1SigUp_au, discInnerEdge1SigDown_au, discOuterEdge_au, discOuterEdge1SigUp_au, discOuterEdge1SigDown_au, sMm_mm, mDisc1km_mEarth, debug=False):
	'''Get uncertainties on all values for the min-mass disc to self 
	stir'''
	
	# Unpack the results
	mDisc_mEarth = discParsForMinDiscMassToSelfStir['minDiscMassToSelfStir_mEarth']
	sMax_km = discParsForMinDiscMassToSelfStir['minSMaxToStirDisc_km']
	sKm_km = discParsForMinDiscMassToSelfStir['sKmToStir_km']
	vFrag_kmPerS = discParsForMinDiscMassToSelfStir['vFragSkm_kmPerS']
									
	# Containers for different variables
	variableValues1SigUp_unit = {}
	variableValues1SigDown_unit = {}
	
	variableValues1SigUp_unit['mDust_mEarth'] = mDustError_mEarth
	variableValues1SigDown_unit['mDust_mEarth'] = mDustError_mEarth
		
	variableValues1SigUp_unit['starMass_MSun'] = starMass1SigUp_MSun
	variableValues1SigDown_unit['starMass_MSun'] = starMass1SigDown_MSun
	
	variableValues1SigUp_unit['starAge_Myr'] = starAge1SigUp_Myr
	variableValues1SigDown_unit['starAge_Myr'] = starAge1SigDown_Myr	
	
	variableValues1SigUp_unit['discInnerEdge_au'] = discInnerEdge1SigUp_au
	variableValues1SigDown_unit['discInnerEdge_au'] = discInnerEdge1SigDown_au

	variableValues1SigUp_unit['discOuterEdge_au'] = discOuterEdge1SigUp_au
	variableValues1SigDown_unit['discOuterEdge_au'] = discOuterEdge1SigDown_au

	variableValues1SigUp_unit['qBig'] = qBigError
	variableValues1SigDown_unit['qBig'] = qBigError
		
	# Regularly-occuring values that appear in many calculations
	C = 2./3. / ((sKm_km*1000.)**-0.36 + sKm_km**1.38) * (-0.36*(sKm_km*1000.)**-0.36 + 1.38*sKm_km**1.38)
	A = (4.*C-qBig+qMed)/(qBig-qMed)
	freqDenominator = (7.-qBig + (4.-qBig)*0.59*A)
	D = 0.59/(qBig-qMed)**2*np.log(mDisc_mEarth/mDisc1km_mEarth)*(mDisc_mEarth/mDisc1km_mEarth)*sKm_km * (np.log(sKm_km)-4.*C/sKm_km)
	a1PlusA2_au = discInnerEdge_au + discOuterEdge_au
	a2MinA1_au = discOuterEdge_au - discInnerEdge_au
	E_mEarthPerKm = mDisc1km_mEarth/sMax_km*(4.-qBig)
	
	# Uncertainties on SMax
	differentialSMaxWRTVariables_kmPerUnit = {
		'starAge_Myr': -sMax_km/starAge_Myr * (1.+.59*A) / freqDenominator,
		'starMass_MSun': -sMax_km/starMass_MSun * (0.5+.79*A) / freqDenominator,
		'discInnerEdge_au': sMax_km/a1PlusA2_au * (2.5 - a1PlusA2_au/a2MinA1_au + 2.56*A) / freqDenominator,
		'discOuterEdge_au': sMax_km/a1PlusA2_au * (2.5 + a1PlusA2_au/a2MinA1_au + 2.56*A) / freqDenominator,
		'qBig': 1./(4.-qBig) * (1.+D) / (np.log(sMax_km)*(1.+D)-3./sMax_km),
		'mDust_mEarth': -sMax_km/mDust_mEarth * (1.+0.59*A) / freqDenominator
		}

	minSMaxToStirDisc1SigUp_km, minSMaxToStirDisc1SigDown_km = GetGaussianAsymmetricUncertainties(variableValues1SigUp_unit, variableValues1SigDown_unit, differentialSMaxWRTVariables_kmPerUnit)

	# Derivatives wrt mDisc1km
	differentialMDisc1kmWRTVariables_mEarthPerUnit = {
		'starAge_Myr': E_mEarthPerKm * differentialSMaxWRTVariables_kmPerUnit['starAge_Myr'],
		'starMass_MSun': E_mEarthPerKm * differentialSMaxWRTVariables_kmPerUnit['starMass_MSun'],
		'discInnerEdge_au': E_mEarthPerKm * differentialSMaxWRTVariables_kmPerUnit['discInnerEdge_au'],
		'discOuterEdge_au': E_mEarthPerKm * differentialSMaxWRTVariables_kmPerUnit['discOuterEdge_au'],
		'qBig': mDisc1km_mEarth*(1./(4.-qBig) - np.log(sMax_km)*differentialSMaxWRTVariables_kmPerUnit['qBig']),
		'mDust_mEarth': mDisc1km_mEarth/mDust_mEarth + E_mEarthPerKm * differentialSMaxWRTVariables_kmPerUnit['mDust_mEarth']
		}
	
	# Uncertainties on SKm
	differentialSKmWRTVariables_kmPerUnit = {
		'starAge_Myr': -0.59*sKm_km/(qBig-qMed)*(1./starAge_Myr + 1./mDisc1km_mEarth*differentialMDisc1kmWRTVariables_mEarthPerUnit['starAge_Myr']),
		'starMass_MSun': -sKm_km/(qBig-qMed)*(0.79/starMass_MSun + 0.59/mDisc1km_mEarth*differentialMDisc1kmWRTVariables_mEarthPerUnit['starMass_MSun']),
		'discInnerEdge_au': sKm_km/(qBig-qMed)*(2.56/a1PlusA2_au - 0.59/mDisc1km_mEarth*differentialMDisc1kmWRTVariables_mEarthPerUnit['discInnerEdge_au']),
		'discOuterEdge_au': sKm_km/(qBig-qMed)*(2.56/a1PlusA2_au - 0.59/mDisc1km_mEarth*differentialMDisc1kmWRTVariables_mEarthPerUnit['discOuterEdge_au']),
		'qBig': 0.59*sKm_km/(qBig-qMed)**2 * np.log(mDisc_mEarth/mDisc1km_mEarth)*(mDisc_mEarth/mDisc1km_mEarth)/mDisc1km_mEarth*differentialMDisc1kmWRTVariables_mEarthPerUnit['qBig'],
		'mDust_mEarth': -0.59*sKm_km/(qBig-qMed)/mDisc1km_mEarth*differentialMDisc1kmWRTVariables_mEarthPerUnit['mDust_mEarth']
		}	
	
	sKmToStir1SigUp_km, sKmToStir1SigDown_km = GetGaussianAsymmetricUncertainties(variableValues1SigUp_unit, variableValues1SigDown_unit, differentialSKmWRTVariables_kmPerUnit)
		
	# Uncertainties on vFrag
	differentialVFragWRTVariables_kmPerSPerUnit = {
		'starAge_Myr': vFrag_kmPerS/sKm_km*C * differentialSKmWRTVariables_kmPerUnit['starAge_Myr'],
		'starMass_MSun': vFrag_kmPerS/sKm_km*C * differentialSKmWRTVariables_kmPerUnit['starMass_MSun'],
		'discInnerEdge_au': vFrag_kmPerS/sKm_km*C * differentialSKmWRTVariables_kmPerUnit['discInnerEdge_au'],
		'discOuterEdge_au': vFrag_kmPerS/sKm_km*C * differentialSKmWRTVariables_kmPerUnit['discOuterEdge_au'],
		'qBig': vFrag_kmPerS/sKm_km*C * differentialSKmWRTVariables_kmPerUnit['qBig'],
		'mDust_mEarth': vFrag_kmPerS/sKm_km*C * differentialSKmWRTVariables_kmPerUnit['mDust_mEarth']
		}	

	vFragSkm1SigUp_kmPerS, vFragSkm1SigDown_kmPerS = GetGaussianAsymmetricUncertainties(variableValues1SigUp_unit, variableValues1SigDown_unit, differentialVFragWRTVariables_kmPerSPerUnit)
	
	# Uncertainties on mDisc
	differentialMDiscToStirWRTVariables_mEarthPerUnit = {
		'starAge_Myr': mDisc_mEarth*(-1./starAge_Myr + 4./vFrag_kmPerS*differentialVFragWRTVariables_kmPerSPerUnit['starAge_Myr'] - 3./sMax_km*differentialSMaxWRTVariables_kmPerUnit['starAge_Myr']),
		'starMass_MSun': mDisc_mEarth*(-0.5/starMass_MSun + 4./vFrag_kmPerS*differentialVFragWRTVariables_kmPerSPerUnit['starMass_MSun'] - 3./sMax_km*differentialSMaxWRTVariables_kmPerUnit['starMass_MSun']),
		'discInnerEdge_au': mDisc_mEarth*(-1./a2MinA1_au + 2.5/a1PlusA2_au + 4./vFrag_kmPerS*differentialVFragWRTVariables_kmPerSPerUnit['discInnerEdge_au'] - 3./sMax_km*differentialSMaxWRTVariables_kmPerUnit['discInnerEdge_au']),
		'discOuterEdge_au': mDisc_mEarth*( 1./a2MinA1_au + 2.5/a1PlusA2_au + 4./vFrag_kmPerS*differentialVFragWRTVariables_kmPerSPerUnit['discOuterEdge_au'] - 3./sMax_km*differentialSMaxWRTVariables_kmPerUnit['discOuterEdge_au']),
		'qBig': mDisc_mEarth*(4./vFrag_kmPerS*differentialVFragWRTVariables_kmPerSPerUnit['qBig'] - 3./sMax_km*differentialSMaxWRTVariables_kmPerUnit['qBig']),
		'mDust_mEarth': mDisc_mEarth*(4./vFrag_kmPerS*differentialVFragWRTVariables_kmPerSPerUnit['mDust_mEarth'] - 3./sMax_km*differentialSMaxWRTVariables_kmPerUnit['mDust_mEarth']), 
		}		

	minDiscMassToSelfStir1SigUp_mEarth, minDiscMassToSelfStir1SigDown_mEarth = GetGaussianAsymmetricUncertainties(variableValues1SigUp_unit, variableValues1SigDown_unit, differentialMDiscToStirWRTVariables_mEarthPerUnit)
	
	# Package the values
	uncertaintiesOnDiscParsForMinDiscMassToSelfStir = {
		'minSMaxToStirDisc1SigUp_km': minSMaxToStirDisc1SigUp_km,
		'minSMaxToStirDisc1SigDown_km': minSMaxToStirDisc1SigDown_km,
		'sKmToStir1SigUp_km': sKmToStir1SigUp_km,
		'sKmToStir1SigDown_km': sKmToStir1SigDown_km,
		'vFragSkm1SigUp_kmPerS': vFragSkm1SigUp_kmPerS,
		'vFragSkm1SigDown_kmPerS': vFragSkm1SigDown_kmPerS,
		'minDiscMassToSelfStir1SigUp_mEarth': minDiscMassToSelfStir1SigUp_mEarth,
		'minDiscMassToSelfStir1SigDown_mEarth': minDiscMassToSelfStir1SigDown_mEarth
		}

	# Debug code
	if debug:
		print('mDisc1km_mEarth: %s' % mDisc1km_mEarth)
		print('A: %s' % A)
		print('C: %s' % C)
		print('D: %s' % D)
		print('freqDenominator: %s' % freqDenominator)
		print()
		print('Differentials:')
		print('WRT SMax:', differentialSMaxWRTVariables_kmPerUnit)
		print('WRT MDisc1km:', differentialMDisc1kmWRTVariables_mEarthPerUnit)
		print('WRT SKm:', differentialSKmWRTVariables_kmPerUnit)
		print('WRT VFrag:', differentialVFragWRTVariables_kmPerSPerUnit)
		print('WRT MDiscToStir:', differentialMDiscToStirWRTVariables_mEarthPerUnit)
		print()
		print('Uncertainties:')					
		print(uncertaintiesOnDiscParsForMinDiscMassToSelfStir)
		
	return uncertaintiesOnDiscParsForMinDiscMassToSelfStir

############################ Print functions ############################
def CheckUserInputsOK():
	'''Check the user inputs are OK. All values should be positive, and 
	all parameters non-zero (although uncertainties can be zero)'''
	
	areUserInputsOK = True
	reasonsUnputsAreBad = []
	
	# All values must be positive and non-zero
	for value in [mStar_mSun, age_Myr, discInnerEdgeApo_au, discOuterEdgePeri_au, mDust_mEarth]:
		if math.isnan(value) or value <= 0:
			areUserInputsOK = False
			reasonsUnputsAreBad.append('All parameters should be non-zero, positive, and not nan (although uncertainties can be zero or nan)')
			break

	# All uncertainties must be positive or zero
	for uncertaintyValue in [mStar1SigUp_mSun, mStar1SigDown_mSun, age1SigUp_Myr, age1SigDown_Myr, 
		discInnerEdgeApo1SigUp_au, discInnerEdgeApo1SigDown_au, discOuterEdgePeri1SigUp_au, 
		discOuterEdgePeri1SigDown_au, mDustError_mEarth]:
		if math.isnan(uncertaintyValue) == False and uncertaintyValue < 0:
			reasonsUnputsAreBad.append('All uncertainties must each be either zero, positive or nan')
			areUserInputsOK = False			
			break
	
	# Disc apo and peri must be defined correctly
	if discOuterEdgePeri_au < discInnerEdgeApo_au:
		reasonsUnputsAreBad.append('Disc inner edge apocentre is greater than outer edge pericentre. This is possible for eccentric, narrow discs, but you will need to modify these values for the program to work. You will need to change discInnerEdgeApo_au and discOuterEdgePeri_au to estimates of the semimajor axes of the innermost and outermost disc particles, respectively. See notes on HD202628 and Fomalhaut C in Pearce et al. 2022 (Appendicies B27 and B65) for suggestions of values to use')
		areUserInputsOK = False
			
	# Warn the user if the inputs are bad
	if areUserInputsOK == False:
		print('***ERROR*** Problem(s) with user inputs:')
		for reasonUnputsAreBad in reasonsUnputsAreBad:
			print('     -%s' % reasonUnputsAreBad)
		print()	
					
	return areUserInputsOK
	
#------------------------------------------------------------------------
def PrintUserInputs():
	'''Print the user inputs'''
	
	print('User inputs:')
	print('     Star mass: %s MSun' % GetValueAndUncertaintyString(mStar_mSun, mStar1SigUp_mSun, mStar1SigDown_mSun))
	print('     System age: %s Myr' % GetValueAndUncertaintyString(age_Myr, age1SigUp_Myr, age1SigDown_Myr))
	print('     Disc inner edge apo: %s au' % GetValueAndUncertaintyString(discInnerEdgeApo_au, discInnerEdgeApo1SigUp_au, discInnerEdgeApo1SigDown_au))
	print('     Disc outer edge peri: %s au' % GetValueAndUncertaintyString(discOuterEdgePeri_au, discOuterEdgePeri1SigUp_au, discOuterEdgePeri1SigDown_au))
	print('     Millimeter dust mass: %s MEarth' % GetValueAndUncertaintyString(mDust_mEarth, mDustError_mEarth, mDustError_mEarth))
	print()

#------------------------------------------------------------------------
def PrintProgramOutputs(discParsAndErrorsForMinDiscMassToSelfStir):
	'''Print the calculation results'''
	
	# Unpack the parameters and uncertainties
	minDiscMassToStirDisc_mEarth = discParsAndErrorsForMinDiscMassToSelfStir['minDiscMassToSelfStir_mEarth']
	minDiscMassToSelfStir1SigUp_mEarth = discParsAndErrorsForMinDiscMassToSelfStir['minDiscMassToSelfStir1SigUp_mEarth']
	minDiscMassToSelfStir1SigDown_mEarth = discParsAndErrorsForMinDiscMassToSelfStir['minDiscMassToSelfStir1SigDown_mEarth']
	minSMaxToStirDisc_km = discParsAndErrorsForMinDiscMassToSelfStir['minSMaxToStirDisc_km']
	minSMaxToStirDisc1SigUp_km = discParsAndErrorsForMinDiscMassToSelfStir['minSMaxToStirDisc1SigUp_km']
	minSMaxToStirDisc1SigDown_km = discParsAndErrorsForMinDiscMassToSelfStir['minSMaxToStirDisc1SigDown_km']
	sKmToStir_m = 1000*discParsAndErrorsForMinDiscMassToSelfStir['sKmToStir_km']
	sKmToStir1SigUp_m = 1000*discParsAndErrorsForMinDiscMassToSelfStir['sKmToStir1SigUp_km']
	sKmToStir1SigDown_m = 1000*discParsAndErrorsForMinDiscMassToSelfStir['sKmToStir1SigDown_km']
	vFragSkm_mPerS = 1000*discParsAndErrorsForMinDiscMassToSelfStir['vFragSkm_kmPerS']
	vFragSkm1SigUp_mPerS = 1000*discParsAndErrorsForMinDiscMassToSelfStir['vFragSkm1SigUp_kmPerS']
	vFragSkm1SigDown_mPerS = 1000*discParsAndErrorsForMinDiscMassToSelfStir['vFragSkm1SigDown_kmPerS']
							
	print('Results:')
	print('     Min. disc mass required for self-stirring: %s MEarth' % GetValueAndUncertaintyString(minDiscMassToStirDisc_mEarth, minDiscMassToSelfStir1SigUp_mEarth, minDiscMassToSelfStir1SigDown_mEarth))
	print('     Min. SMax required for self-stirring: %s km' % GetValueAndUncertaintyString(minSMaxToStirDisc_km, minSMaxToStirDisc1SigUp_km, minSMaxToStirDisc1SigDown_km))
	print('     Largest colliding particle (SKm) in min.-mass disc: %s m' % GetValueAndUncertaintyString(sKmToStir_m, sKmToStir1SigUp_m, sKmToStir1SigDown_m))
	print('     Fragmentation speed of largest colliding particle in min.-mass disc: %s m/s' %GetValueAndUncertaintyString(vFragSkm_mPerS, vFragSkm1SigUp_mPerS, vFragSkm1SigDown_mPerS))
	
	print()
	
#------------------------------------------------------------------------
def GetValueAndUncertaintyString(value, err1SigUp, err1SigDown):
	'''Get a string neatly showing the value and its uncertainties'''
	
	# The orders of the largest and smallest values that should be 
	# written in non-SI notation
	minOrderForNonSINotation = -3
	maxOrderForNonSINotation = 3
		
	# If the value is non-zero, and neither it nor its uncertainties are 
	# nans, proceed. Some of the functions below would otherwise fail
	if math.isnan(value) == False and math.isnan(err1SigUp) == False and math.isnan(err1SigDown) == False and value > 0:

		# Errors quoted to 1 sig fig
		err1SigUpRounded = abs(RoundNumberToDesiredSigFigs(err1SigUp, 1))
		err1SigDownRounded = abs(RoundNumberToDesiredSigFigs(err1SigDown, 1))

		# Get the order of the value and the smallest uncertainty
		orderOfValue = GetBase10OrderOfNumber(value)
		orderOfSmallestError = GetBase10OrderOfNumber(min(err1SigUpRounded, err1SigDownRounded))

		# Round the value to the correct number of significant figures,
		# such that the final figure is at the order of the error
		sigFigsToRoundValueTo = max(orderOfValue - orderOfSmallestError + 1, 1)
		valueRounded = RoundNumberToDesiredSigFigs(value, sigFigsToRoundValueTo)
		orderOfRoundedValue = GetBase10OrderOfNumber(valueRounded)

		# If the rounded value has gone up an order, will round value 
		# to an extra significant figure (e.g. 0.099 +/- 0.03 -> 0.10 +/- 0.03)
		if orderOfRoundedValue > orderOfValue:
			sigFigsToRoundValueTo += 1
		
		# If the value is very small or large, divide it by its order, 
		# and later quote order in string. Use rounding function to 
		# remove rounding errors, and note that uncertainties are 
		# always quoted to 1 sig fig
		if orderOfRoundedValue > maxOrderForNonSINotation or orderOfRoundedValue < minOrderForNonSINotation:
			wasPowerAdjustmentDone = True
			
			#valueRounded /= 10**orderOfRoundedValue
			valueRounded = RoundNumberToDesiredSigFigs(valueRounded/10**orderOfRoundedValue, sigFigsToRoundValueTo)
			err1SigUpRounded = RoundNumberToDesiredSigFigs(err1SigUpRounded/10**orderOfRoundedValue, 1)
			err1SigDownRounded  = RoundNumberToDesiredSigFigs(err1SigDownRounded/10**orderOfRoundedValue, 1)
			
			orderOfRoundedValueAfterPowerAdjust = GetBase10OrderOfNumber(valueRounded)
			orderOfSmallestRoundedErrorAfterPowerAdjust = GetBase10OrderOfNumber(min(err1SigUpRounded, err1SigDownRounded))
			
		else:
			wasPowerAdjustmentDone = False
			orderOfRoundedValueAfterPowerAdjust = orderOfRoundedValue
			orderOfSmallestRoundedErrorAfterPowerAdjust = orderOfSmallestError

		# If all significant figures of both the value and 
		# uncertainties are to the left of the decimal point, the value
		# and uncertainties are integers
		numberOfValueFiguresLeftOfDecimalPoint = max(orderOfRoundedValueAfterPowerAdjust + 1, 0)
		numberOfErrorFiguresLeftOfDecimalPoint = orderOfSmallestRoundedErrorAfterPowerAdjust + 1

		if numberOfValueFiguresLeftOfDecimalPoint - sigFigsToRoundValueTo >= 0:
			valueRounded = int(valueRounded)
			err1SigUpRounded = int(err1SigUpRounded)
			err1SigDownRounded = int(err1SigDownRounded)

		# Convert the value to a string. If there are value figures to 
		# the right of the decimal point, and the final one(s) should 
		# be zero, append zeros to the string 
		valueRoundedString = str(valueRounded)
		
		numberOfValueFiguresNeededRightOfDecimalPoint = max(sigFigsToRoundValueTo - (orderOfRoundedValueAfterPowerAdjust+1), 0)
		
		if numberOfValueFiguresNeededRightOfDecimalPoint > 0:

			while True:
				indexOfPointInString = valueRoundedString.index('.')
				numberOfValueFiguresRightOfDecimalPointInString = len(valueRoundedString[indexOfPointInString+1:])
									
				if numberOfValueFiguresRightOfDecimalPointInString == numberOfValueFiguresNeededRightOfDecimalPoint:
					break
				
				valueRoundedString += '0'				
		
		# If errors are symmetric
		if err1SigUpRounded == err1SigDownRounded:
			if wasPowerAdjustmentDone:
				valueAndUncertaintyString = '(%s +/- %s) * 10^%s' % (valueRoundedString, err1SigUpRounded, orderOfRoundedValue)

			else:
				valueAndUncertaintyString = '%s +/- %s' % (valueRoundedString, err1SigUpRounded)

		# Otherwise errors are asymmetric
		else:
			if wasPowerAdjustmentDone:
				valueAndUncertaintyString = '(%s +%s -%s) * 10^%s' % (valueRoundedString, err1SigUpRounded, err1SigDownRounded, orderOfRoundedValue)	
			
			else:
				valueAndUncertaintyString = '%s +%s -%s' % (valueRoundedString, err1SigUpRounded, err1SigDownRounded)
					
	# Otherwise, if both uncertainties are nan, then only quote the value
	elif math.isnan(err1SigUp) and math.isnan(err1SigDown):
		valueAndUncertaintyString = value

	# Otherwise either the value is zero or nan, or only one error is nan
	else:
		valueAndUncertaintyString = '%s +%s -%s' % (value, err1SigUp, err1SigDown)

	return valueAndUncertaintyString
	
################################ Program ################################
print()

# Print the user inputs
PrintUserInputs()
	
# Check user inputs fine
areUserInputsOK = CheckUserInputsOK()

# Continue if the user inputs are OK
if areUserInputsOK:

	# Get the minimum disc mass to stir the disc within the star lifetime
	discParsAndErrorsForMinDiscMassToSelfStir = GetDiscParsAndErrorsForMinDiscMassToSelfStir_mEarthAndKm(mDust_mEarth, mDustError_mEarth, age_Myr, age1SigUp_Myr, age1SigDown_Myr, mStar_mSun, mStar1SigUp_mSun, mStar1SigDown_mSun, discInnerEdgeApo_au, discInnerEdgeApo1SigUp_au, discInnerEdgeApo1SigDown_au, discOuterEdgePeri_au, discOuterEdgePeri1SigUp_au, discOuterEdgePeri1SigDown_au)

	minDiscMassToStirDisc_mEarth = discParsAndErrorsForMinDiscMassToSelfStir['minDiscMassToSelfStir_mEarth']
	minSMaxToStirDisc_km = discParsAndErrorsForMinDiscMassToSelfStir['minSMaxToStirDisc_km']
		
	# Check numerical fitting has converged; the disc minimum 
	# disc mass required to self-stir should equal the disc 
	# mass at that value of SMax
	discMassAtMinSMax_mEarth, sKmAtMinSMaxToStirDisc_kmAlt, mDisc1km_mEarth = GetDiscMassAndSkmFromSMax_mEarthAndKm(minSMaxToStirDisc_km, mDust_mEarth, age_Myr, mStar_mSun, 0.5*(discInnerEdgeApo_au+discOuterEdgePeri_au))
	allowedFractionalMassTolerance = 1e-6
	fracDifferenceBetweenMasses = abs(discMassAtMinSMax_mEarth/minDiscMassToStirDisc_mEarth - 1)

	if fracDifferenceBetweenMasses > allowedFractionalMassTolerance:
		print('     ### WARNING: %s min self-stirring mass may not have converged. Min SMax to self-stir: %s km. Self-stirring mass: %s mEarth. Disc mass at that SMax: %s mEarth. Frac difference: %s ###' % (targetName, minSMaxToStirDisc_km, minDiscMassToStirDisc_mEarth, discMassAtMinSMax_mEarth, fracDifferenceBetweenMasses))

	# Print the results
	PrintProgramOutputs(discParsAndErrorsForMinDiscMassToSelfStir)
	
#########################################################################

