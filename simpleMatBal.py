"""

Simple Material Balance Solver for freeColumn Beta

Author: Piero Wemyss
Created: September 21, 2024

"""

import numpy as np
from dict2struct import dict2struct

def simpleMatBal(zF,F,E2F,xE,specs):

	solved = dict2struct()
	solved.F = F
	solved.zF = zF
	
	FR_LK = specs.FR_LK
	NK_spec = specs.NK_spec
	
	if opts.massBal == 1:
	
		if LK_ind == 1:
		    if opts.extract == 1:
		    fD = FR_LK*zF(LK_ind) + (1-FR_LK)*zF(LK_ind+1) + (length(zF)-(LK_ind+2))*NK_spec
		    else:
		    fD = FR_LK*zF(LK_ind) + (1-FR_LK)*zF(LK_ind+1) + np.sum(zF(LK_ind+2:end))*NK_spec
		   
		else:
		    fD = FR_LK*zF(LK_ind) + (1-FR_LK)*zF(LK_ind+1) + np.sum(zF(1:LK_ind-1)) + (length(zF)-(LK_ind+2))*NK_spec
		    fD = (1-NK_spec)*np.sum(zF(1:LK_ind-1)) + FR_LK*zF(LK_ind) + (1-FR_LK)*zF(LK_ind+1) + sum(zF(LK_ind+2:end))*NK_spec
		
		xD(1:LK_ind-1) = (1-NK_spec)*zF(1:LK_ind-1)/fD
		xD(LK_ind) = FR_LK*zF(LK_ind)/fD
		xD(LK_ind+1) = (1-FR_LK)*zF(LK_ind+1)/fD
		if opts.extract == 1:
		    xD(LK_ind+2:length(zF)) = NK_spec*ones(1,length(zF(LK_ind+2:end)))
		    xD(1:LK_ind+1) = xD(1:LK_ind+1) - np.sum(xD(LK_ind+2:length(zF)))/length(xD(1:LK_ind+1))
		else:
		    xD(LK_ind+2:length(zF)) = NK_spec*zF(LK_ind+2:end)/fD
		
		# sumFunc = @(scal) 1 - np.sum(scal.*xD)
		# scalN = fsolve(sumFunc,1,opts.fsolve)
		# fprintf('\n\nMake sure this value equals 1: %4.4f\n\n', scalN)
		# xD = scalN.*xD
		
		D = fD*F
		
		fB = 1 - fD
		
		xB(1:LK_ind-1) = NK_spec*zF(1:LK_ind-1)/fB
		xB(LK_ind) = (1-FR_LK)*zF(LK_ind)/fB
		xB(LK_ind+1) = (FR_LK)*zF(LK_ind+1)/fB
		xB(LK_ind+2:length(zF)) = (1-NK_spec)*zF(LK_ind+2:length(zF))/fB
		
		# sumFunc = @(scal) 1 - np.sum(scal.*xB)
		# scalN = fsolve(sumFunc,1,opts.fsolve)
		# fprintf('\nMake sure this value equals 1: %4.4f\n\n', scalN)
		# xB = scalN.*xB
		
		# xB(LK_ind:end) = xB(LK_ind:end) - np.sum(xB(1:LK_ind-1))/length(xB(LK_ind:end))
		
		B = fB*F
		
		if opts.extract == 1:
		
		    E = E2F*F
		
		    xB = (B*xB + E*xE)/(B + E)
		    B = E + B
		
		
		solved.xD = xD
		solved.D = D
		solved.xB = xB
		solved.B = B
		
		# disp(np.sum(F*zF - (D*xD + B*xB)))
		# disp(F-D-B)
		
		# else:
		# 
		# xD = specs.xD
		# xB = specs.xB
		# 
		# D = F*(zF(LK_ind)-xB(LK_ind))/(xD(LK_ind)-xB(LK_ind))
		# B = F - D
		# % B = F*(zF(LK_ind)-xD(LK_ind))/(xB(LK_ind)-xD)
		# 
		# solved.xD = xD
		# solved.D = D
		# solved.xB = xB
		# solved.B = B
		# 
		#
	
	else:
	
		if LK_ind == 1:
		    fD = FR_LK*zF(LK_ind) + (1-FR_LK)*zF(LK_ind+1) + (length(zF)-(LK_ind+2))*NK_spec
		else:
		    fD = FR_LK*zF(LK_ind) + (1-FR_LK)*zF(LK_ind+1) + np.sum(zF(1:LK_ind-1)) + (length(zF)-(LK_ind+2))*NK_spec
		
		xD(1:LK_ind-1) = zF(1:LK_ind-1)/fD
		xD(LK_ind) = FR_LK*zF(LK_ind)/fD
		xD(LK_ind+1) = (1-FR_LK)*zF(LK_ind+1)/fD
		xD(LK_ind+2:length(zF)) = NK_spec*ones(1,length(zF)-length(xD))
		xD(1:LK_ind+1) = xD(1:LK_ind+1) - np.sum(xD(LK_ind+2:length(zF)))/length(xD(1:LK_ind+1))
		solved.xD = xD
		
		D = fD*F
		solved.D = D
		
		fB = 1 - fD
		
		xB(1:LK_ind-1) = NK_spec*ones(1,LK_ind-1)
		xB(LK_ind) = (1-FR_LK)*zF(LK_ind)/fB
		xB(LK_ind+1) = (FR_LK)*zF(LK_ind+1)/fB
		xB(LK_ind+2:length(zF)) = zF(length(xB)+1:length(zF))/fB
		xB(LK_ind:end) = xB(LK_ind:end) - np.sum(xB(1:LK_ind-1))/length(xB(LK_ind:end))
		
		B = fB*F
		
		if opts.extract == 1:
		
		    E = E2F*F
		
		    xB = (B*xB + E*xE)/(B + E)
		    B = E + B
		
		
		solved.B = B
		solved.xB = xB
	
	
	
	if np.sum(xD) > 1.0001:
	    warning(['Distillate fraction does not sum to 1. Sum is ',num2str(np.sum(xD))])
	if np.sum(xD) < 0.9999:
	    warning(['Distillate fraction does not sum to 1. Sum is ',num2str(np.sum(xD))])
	if np.sum(xB) > 1.0001:
	    warning(['Bottoms fraction does not sum to 1. Sum is ',num2str(np.sum(xB))])
    if np.sum(xB) < 0.9999:
	    warning(['Bottoms fraction does not sum to 1. Sum is ',num2str(np.sum(xB))])

    return solved
