% loss_sqd - conjugate of weighted squared loss function
%
% Syntax:
% [floss, gloss, hloss, hmin]=loss_sqd(aa, bb, weight)
% 
% Copyright(c) 2009 Ryota Tomioka
% This software is distributed under the MIT license. See license.txt
function varargout = loss_sqdw(aa, bb, weight)

gloss = aa./weight-bb;
floss = 0.5*sum(weight.*gloss.^2)-0.5*sum(weight.*bb.^2);
hloss = spdiag(1./weight);
hmin  = 1/max(weight);
  
if nargout<=3
  varargout = {floss, gloss, hmin};
else
  varargout = {floss, gloss, hloss, hmin};
end

