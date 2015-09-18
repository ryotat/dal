function out=fAwmat(A, W)

[m,n]=size(A);
[t,n,k]=size(W);

out=zeros(m*t, k);
for ii=1:k
  out(:,ii) = vec(A*W(:,:,ii)');
end