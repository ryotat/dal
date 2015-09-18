function out=fAwmatt(A, W)

[m,n]=size(A);
[~,t,k]=size(W);

out=zeros(t*n, k);
for ii=1:k
  out(:,ii) = vec(W(:,:,ii)'*A);
end