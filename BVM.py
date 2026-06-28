"""

Author: Piero Wemyss
Created: September 12, 2024

Boundary Value Method for Simple Extractive Columns and
Simple Columns

"""

import numpy as np
import nifco as ni
from simpleMatBal import simpleMatBal
from dict2struct import dict2struct
import warnings


def boundValMethod(zF,F,r,q,E2F,xE,specs,LK_ind,P,comps,opts):

    # Check input
    if np.sum(zF) > 1.001 or np.sum(zF) < 0.999:
        warnings.warn(['Feed composition does not sum to 1. Sum is ', np.sum(zF)])

    # Solve Overall Mass Balance

    column = dict2struct()
    column.F = F
    column.zF = zF

    TMB = simpleMatBal(zF,F,E2F,xE)
    xD = TMB.xD
    D = TMB.D
    xB = TMB.xB
    B = TMB.B
    E = E2F*F

    column.xD
    column.D
    column.xB
    column.B
    column.xE
    column.E

    # Compute Boilup Ratio
    s = (D/B)*(r+q) - (1-q)
    column.q = q
    column.r = r
    column.s = s

    # Rectifying Section
    n = opts.n

    xRect = np.array([1,(20+n)/opts.efficiency])
    yRect = np.array([1,(20+n)/opts.efficiency])

    yRect[0,:] = xD
    
    #point where I left off

    [~,distInd] = max(xD)
    condAntC = fetchProps(1,comps(distInd))
    T0 = (condAntC.antoine(2)/(condAntC.antoine(1)-log10(P)) - condAntC.antoine(3))
    T0 = fsolve(@(T) antoineCalc(T,comps(distInd),opts.antMethod)-P,T0,opts.fsolve)
    
    X0 = [xD, T0]
    
    for i = 1:(20+n)/opts.efficiency
    
        [rect,~,rectFlag(i)] = fsolve(@(X) rectVLE(X,yRect(i,:),P,comps,opts),X0,opts.fsolve)
        xRect(i,:) = rect(1:length(comps))
        # if i == 1
        #     % xRect(i,:) = opts.efficiency*(xRect(i,:)-xD) + xD
        #     xRect(i,:) = xRect(i,:)
        # else
        #     xRect(i,:) = opts.efficiency*(xRect(i,:)-xRect(i-1,:)) + xRect(i-1,:)
        # end
        Trect(i) = rect(length(comps)+1)
        yRect(i+1,:) = (r/(r+1))*xRect(i,:) + xD/(r+1)
        yRect(i+1,:) = opts.efficiency*(yRect(i+1,:)-yRect(i,:)) + yRect(i,:)
        X0 = [xRect(i,:), Trect(i)]
    
    end
    
    yRect = yRect(1:end-1,:)
    
    if min(rectFlag) < 1
        warning('Caution: One rectifying stage may not have solved correctly.')
    end
    
    # end
    #% Stripping Section
    
    rebAntC = fetchProps(1,comps(LK_ind+1))
    T0 = rebAntC.antoine(2)/(rebAntC.antoine(1)-log10(P)) - rebAntC.antoine(3)
    T0 = fsolve(@(T) antoineCalc(T,comps(LK_ind+1),opts.antMethod)-P,T0,opts.fsolve)
    
    xStrip(1,:) = xB
    Y0 = [xB, T0]
    
    for i = 1:(20+n)/opts.efficiency
    
        [strip,~,stripFlag(i)] = fsolve(@(Y) stripVLE(xStrip(i,:),Y,P,comps,opts),Y0,opts.fsolve)
    
        # if stripFlag(i) < 1
        #     opts.reSolve = 1
        #     lb = [zeros(1,length(yStrip(i-1,:))), Trect(1)]
        #     ub = [ones(1,length(yStrip(i-1,:))), Tstrip(i-1)+1]
        #     Aeq = [ones(1,length(yStrip(i-1,:))), 0]
        #     beq = 1
        # 
        #     % rng default
        #     % gs = GlobalSearch
        #     % problem = createOptimProblem('fmincon','x0',Y0,'objective',...
        #     %     @(Y) stripVLE(xStrip(i,:),Y,P,comps,opts),'lb',lb,...
        #     %     'ub',ub,'Aeq',Aeq,'beq',beq)
        #     % [strip,stripResid] = run(gs,problem)
        #     [strip,stripResid] = fminimax(@(Y) stripVLE(xStrip(i,:),Y,P,comps,opts),Y0,[],[],Aeq,beq,lb,ub)
        #     opts.reSolve = 0
        # 
        #     if stripResid > 1e-2
        #         warning(['Stripping stage ',num2str(i),' unsolved with residual of ',num2str(stripResid)])
        #     end
        # end
    
        yStrip(i,:) = strip(1:length(comps))
        # if i == 1
        #     yStrip(i,:) = yStrip(i,:)
        # else
        #     yStrip(i,:) = opts.efficiency*(yStrip(i,:)-yStrip(i-1,:)) + yStrip(i-1,:)
        # end
        Tstrip(i) = strip(length(comps)+1)
        if opts.extract == 0
            xStrip(i+1,:) = (yStrip(i,:) + xB/s)/((s+1)/s)
        elseif opts.extract == 1
            xStrip(i+1,:) = (yStrip(i,:) + ((1+xE.*E/B)/s).*xB)./(((s+1)+xE.*E/B)/s)
        end
        xStrip(i+1,:) = opts.efficiency*(xStrip(i+1,:)-xStrip(i,:)) + xStrip(i,:)
        Y0 = [yStrip(i,:), Tstrip(i)]
    
    end
    
    if min(stripFlag) < 1
        warning('Caution: One stripping stage may not have solved correctly.')
    end
