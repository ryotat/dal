% dalsqds - DAL with squared loss and the dual spectral norm
%           (trace norm) regularization
%           This implementation only keeps the factors of the
%           solution and never forms the full matrix. This only
%           works for the quasi-Newton solver.
%
% Overview:
%  Solves the optimization problem:
%   ww = argmin 0.5||A*x-bb||^2 + lambda*||w||_DS
%
%   where ||w||_DS = sum(svd(w)) 
%
% Syntax:
%  [ww,bias,status]=dalsqds(ww, bias, A, yy, lambda, <opt>)
%
% Inputs:
%  ww     : initial solution ([nn,1])
%  A      : the design matrix A ([mm,nn]) or a cell array {fA, fAT, mm, nn}
%           where fA and fAT are function handles to the functions that
%           return A*x and A'*x, respectively, and mm and nn are the
%           numbers of rows and columns of A.
%  yy     : the target label vector (-1 or +1) ([mm,1])
%  lambda : the regularization constant
%  <opt>  : list of 'fieldname1', value1, 'filedname2', value2, ...
%   stopcond : stopping condition, which can be
%              'pdg'  : Use relative primal dual gap (default)
%              'fval' : Use the objective function value
%           (see dal.m for other options)
% Outputs:
%  ww     : the final solution ([nn,1])
%  status : various status values
%
% Example:
% n = [640 640]; r = 6; m = 10*r*sum(n);
% w0=randsparse(n,'rank',r);
% ind=randperm(prod(n)); ind=ind(1:m)';
% [I,J]=ind2sub(n, ind);
% yy=w0(ind)+0.01*randn(m,1);
% A={@(X)fobsX(X,I,J), @(aa)fobsXadj(aa,I,J,n(1),n(2)), m, prod(n)};
% lambda=0.1*norm(full(A{2}(yy)));
% [ww,stat]=dalsqds(zeros(n),A,yy,lambda,'solver','qn','eta',1);
% W=ww.U*diag(ww.ss)*ww.V';
%
% Copyright(c) 2009 Ryota Tomioka
% This software is distributed under the MIT license. See license.txt

function [ww,status]=dalsqds(ww, A, yy, lambda, varargin)

opt=propertylist2struct(varargin{:});
opt=set_defaults(opt,'solver','qn',...
                     'stopcond','pdg',...
                     'blks',[]);

if ~strcmp(opt.solver, 'qn')
  error(['This version only works with solver=qn. Check out the ' ...
         'master branch to use cg.']);
end

if isempty(opt.blks)
  [R,C]=size(ww);
  opt.blks=[R,C];
  ww=struct('U',zeros(R,10),...
          'ss',zeros(10,1),...
          'V',zeros(C,10),...
          'D', []);
end

prob.floss    = struct('p',@loss_sqp,'d',@loss_sqd,'args',{{yy}});
prob.fspec    = @(ww)ds_spec(ww,opt.blks);
prob.dnorm    = @(ww)ds_dnorm(ww,opt.blks);
prob.obj      = @objdalds;
prob.softth   = @ds_softth;
prob.stopcond = opt.stopcond;
prob.ll       = -inf*ones(size(yy));
prob.uu       = inf*ones(size(yy));
prob.Ac       =[];
prob.bc       =[];
prob.info     = struct('blks',opt.blks,'nsv',5*ones(1,size(opt.blks,1)));

if isequal(opt.solver,'cg')
  prob.hessMult = @hessMultdalds;
end

if isnumeric(A)
  A = A(:,:);
  [mm,nn]=size(A);
  At=A';
  fA = struct('times',@(x)A*x,...
              'Ttimes',@(x)At*x,...
              'slice', @(I)A(:,I));
  clear At;
elseif iscell(A)
  mm = A{3};
  nn = A{4};
  fAslice = @(I)A{1}(sparse(I,1:length(I),ones(length(I),1), nn, length(I)));
  fA = struct('times',A{1},...
              'Ttimes',A{2},...
              'slice',fAslice);
else
  error('A must be either numeric or cell {@(x)A*x, @(y)(A''*y), mm, nn}');
end

prob.mm       = mm;
prob.nn       = nn;

[ww,uu,status]=dal(prob,ww,[],fA,[],lambda,opt);



