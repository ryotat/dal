function [fp, tp]=loss_roccurve(yy, out)

nn=length(out);
np=sum(yy>0);
[ss,ix]=sort(-out);

[uq,I,J]=unique(ss);

tp=zeros(length(I),1);
fp=zeros(length(I),1);

for ii=1:length(I)
  tp(ii)=sum(yy(ix(1:I(ii)))>0)/np;
  fp(ii)=sum(yy(ix(1:I(ii)))<=0)/(nn-np);
end

tp=[0; tp];
fp=[0; fp];