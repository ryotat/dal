% dalsqglrow - DAL with squared loss and row-wise grouped L1 regularization
%
% Overview:
%  Solves the optimization problem:
%   xx = argmin 0.5||A*X-Y||_F^2 + lambda*||X||_G1
%  where
%   ||x||_G1 = sum(sqrt(sum(X'.^2)))
%  (row-wise grouped L1 norm)
%
% Syntax:
%  [xx,status]=dalsqgl(xx0, A, Y, lambda, <opt>)
%
% Inputs:
%  xx0    : initial solution ([n, t])
%  A      : the design matrix A ([m,n])
%  Y      : the target matrix ([m,t])
%  lambda : the regularization constant
%  <opt>  : list of 'fieldname1', value1, 'filedname2', value2, ...
%   stopcond : stopping condition, which can be
%              'pdg'  : Use relative primal dual gap (default)
%              'fval' : Use the objective function value
%           (see dal.m for other options)
% Outputs:
%  xx     : the final solution ([n,t])
%  status : various status values
%
% Example:
% m=200; n=10000; t=100; k=round(0.01*n); A=randn(m,n);
% w0=randsparse([t,n],k)'; 
% Y=A*w0+0.01*randn(m,t);
% lambda=0.1*max(sqrt(sum((Y'*A).^2)))
% [ww,stat]=dalsqglrow(zeros(n,t), A, Y, lambda);
%
% [fp,tp]=loss_roccurve(2*(sum(w0.^2,2)>0)-1, sum(ww.^2,2));
% plot(fp, tp); xlabel('FPR'); ylabel('TPR');
%
% Copyright(c) 2009 Ryota Tomioka
% This software is distributed under the MIT license. See license.txt

function [ww,status]=dalsqglrow(ww, A, Y, lambda, varargin)

opt=propertylist2struct(varargin{:});
opt=set_defaults(opt,'solver','cg',...
                     'stopcond','pdg');

[n,t]=size(ww); ww=ww';
[m,n]=size(A);

opt.blks=t*ones(1,n);

prob.floss    = struct('p',@loss_sqp,'d',@loss_sqd,'args',{{Y(:)}});
prob.fspec    = @(ww)gl_spec(ww,opt.blks);
prob.dnorm    = @(ww)gl_dnorm(ww,opt.blks);
prob.obj      = @objdalgl;
prob.softth   = @gl_softth;
prob.stopcond = opt.stopcond;
prob.ll       = -inf*ones(m*t,1);
prob.uu       = inf*ones(m*t,1);
prob.Ac       =[];
prob.bc       =[];
prob.info     = struct('blks',opt.blks);

if isequal(opt.solver,'cg')
  prob.hessMult = @hessMultdalglrow;
end

if isequal(opt.stopcond,'fval')
  opt.feval = 1;
end

fA=@(w)fAwmat(A,reshape(w,[t,n,size(w,2)]));
fAT=@(aa)fAwmatt(A, reshape(aa,[m,t,size(aa,2)]));
fA = struct('times',fA,...
            'Ttimes',fAT,...
            'num',A);

prob.mm       = m*t;
prob.nn       = n*t;

[ww,uu,status]=dal(prob,ww(:),[],fA,[],lambda,opt);

ww=reshape(ww, [t,n])';

