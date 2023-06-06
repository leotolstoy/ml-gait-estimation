function bezierCoeffs3P = returnBezierEval3P_sL_incline(phase,stepLength,incline)

 inclineFuncs = returnBezierLinear(incline);
 stepLengthFuncs = returnBezierQuadratic(stepLength);
 phaseFuncs = returnBezierCubic(phase);
 
 numInclineFuncs = length(inclineFuncs);
 numStepLengthFuncs = length(stepLengthFuncs);
 numPhaseFuncs = length(phaseFuncs);
    
 bezierCoeffs3P = zeros(1,numInclineFuncs*numStepLengthFuncs*numPhaseFuncs);
 N=1;
 
 for ii = 1:numInclineFuncs
     inclineFunc = inclineFuncs(ii);
     
     for jj = 1:numStepLengthFuncs
         stepLengthFunc = stepLengthFuncs(jj);
         
         for kk = 1:numPhaseFuncs
             phaseFunc = phaseFuncs(kk);
             bezierCoeffs3P(N) = inclineFunc * stepLengthFunc * phaseFunc;
             N = N + 1;
         end
         
     end
     
 end
 
%  bezierCoeffs3P
%  
%  bezierCoeffs3P = [(incline).*stepLength.*(1-phase).^3,...
%      (incline).*stepLength.*3*(1-phase).^2.*phase,...
%      (incline).*stepLength.*3*(1-phase).*(phase).^2,...
%      (incline).*stepLength.*(phase).^3,...
%      (incline).*(1 - stepLength).*(1-phase).^3,...
%      (incline).*(1 - stepLength).*3*(1-phase).^2.*phase,...
%      (incline).*(1 - stepLength).*3*(1-phase).*(phase).^2,...
%      (incline).*(1 - stepLength).*(phase).^3,...
%      (1 - incline).*stepLength.*(1-phase).^3,...
%      (1 - incline).*stepLength.*3*(1-phase).^2.*phase,...
%      (1 - incline).*stepLength.*3*(1-phase).*(phase).^2,...
%      (1 - incline).*stepLength.*(phase).^3,...
%      (1 - incline).*(1 - stepLength).*(1-phase).^3,...
%      (1 - incline).*(1 - stepLength).*3*(1-phase).^2.*phase,...
%      (1 - incline).*(1 - stepLength).*3*(1-phase).*(phase).^2,...
%      (1 - incline).*(1 - stepLength).*(phase).^3]
% pause
end