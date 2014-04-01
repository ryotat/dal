% loss_sqpw - weighted squared loss function
%
% Copyright(c) 2009 Ryota Tomioka
% This software is distributed under the MIT license. See license.txt
function [floss, gloss]=loss_sqpw(zz, bb, weight)

gloss = weight.*(zz-bb);
floss = 0.5*sum(weight.*(zz-bb).^2);