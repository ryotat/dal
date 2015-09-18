% hessMultdalglrow - function that computes H*x for DAL with the
%                    row-wise grouped L1 regularization
%
% Copyright(c) 2009 Ryota Tomioka
% This software is distributed under the MIT license. See license.txt

function yy = hessMultdalglrow(xx, A, eta, Hinfo)

blks =Hinfo.blks;
hloss=Hinfo.hloss;
I    =Hinfo.I;
vv   =Hinfo.vv;
nm   =Hinfo.nm;
lambda=Hinfo.lambda;
[m,n] =size(A.num);
t =size(xx,1)/m;
% p = size(xx,2);

yy = hloss*xx;

nn=sum(blks);

AI=A.num(:,I);
xk=reshape(xx, [m, t])'*AI;
ff=lambda./nm(I);
vv=reshape(vv, [t, n]);
vn=bsxfun(@mrdivide, vv(:,I), nm(I));
vnxk = sum(xk.*vn);
RF=bsxfun(@mtimes, xk, 1-ff)+bsxfun(@mtimes, vn, vnxk.*ff);
yy = yy + eta(1)*vec(AI*RF');

%cumblks=[0,cumsum(blks)];
%for kk=1:length(I)
%  jj=I(kk);
%  bsz=blks(jj);
%  J=cumblks(jj)+(1:bsz);
%  vn=vv(J)/nm(jj);

%  ff=lambda/nm(jj);
%  Ajj=A.num(:,jj);
%  xk=xx'*Ajj;
%  yy = yy + eta(1)*kron((1-ff)*xk+ff*(vn'*xk)*vn, Ajj);
%end

B=Hinfo.B;
if ~isempty(B)
  yy = yy + eta(2)*(B*(B'*xx));
end
