function zz=fobsX(X,I,J)

n=length(I);

zz = (X.U(I,:).*X.V(J,:))*X.ss;

% $$$ Xu=X.U;
% $$$ Yv=X.V';
% $$$ zz=zeros(n,1);
% $$$ for ii=1:n
% $$$   zz(ii)=sum(Xu(I(ii),:)'.*X.ss.*Yv(:,J(ii)));
% $$$   
% $$$ % $$$   if ~isempty(X.D)
% $$$ % $$$     zz(ii)=zz(ii)+X.D(I(ii),J(ii));
% $$$ % $$$   end
% $$$ end

