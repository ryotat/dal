function y=multlrt(x,W)

y=W.V*(diag(W.ss)*(W.U'*x));

if ~isempty(W.D)
  y=y+W.D'*x;
end
