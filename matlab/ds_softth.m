% ds_softth - soft threshold function for DS regularization
%
% Copyright(c) 2009 Ryota Tomioka
% This software is distributed under the MIT license. See license.txt

function [vv,ss,info]=ds_softth(vv,lambda,info)

ss=zeros(sum(min(info.blks,[],2)),1);

ixs=0;
ixv=0;

if isnumeric(vv)
for kk=1:size(info.blks,1)
  blk=info.blks(kk,:);
  I=ixv+(1:blk(1)*blk(2));
  ixv=I(end);
  J=ixs+(1:min(blk));
  ixs=J(end);
  vvk = reshape(vv(I),blk);
  if strcmp(info.solver,'cg')
    [U,S,V]=svd(vvk);
  else
    [U,S,V]=svdmaj(vvk, lambda, 'kinit', info.nsv(kk)+1);
  end
  dd=diag(S);
  K=find(dd>lambda);
  ssk=dd(K)-lambda;
  ss(J(1:length(ssk)))=ssk;
  vvk=U(:,K)*diag(ssk)*V(:,K)';
  vv(I)=vvk(:);
  info.nsv(kk)=length(ssk);
  
  info.U{kk}=U;
  info.S{kk}=S;
  info.V{kk}=V;
end
else
  R=info.blks(1); C=info.blks(2);
  if strcmp(info.solver,'cg') || info.nsv==min(info.blks)
    [U,S,V]=svd(vv.U*diag(vv.ss)*vv.V'+vv.D);
  else
    vcell={@(x)multlr(x,vv),@(x)multlrt(x,vv),R,C};
    [U,S,V]=svdmaj(vcell, lambda, 'kinit', info.nsv+1);
  end
  dd=diag(S);
  K=find(dd>lambda);
  ss=dd(K)-lambda;
  if length(K)>0
    vv=struct('U',U(:,K),...
                    'ss',ss,...
                    'V',V(:,K),...
                    'D',[]);
  else
    vv=struct('U',zeros(R,1),'ss',0,'V',zeros(C,1),'D',[]);
  end
  info.nsv=length(ss);
  
  info.U{1}=U;
  info.S{1}=S;
  info.V{1}=V;
end