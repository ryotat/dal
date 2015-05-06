function y=multlr(x,W)

y=W.U*(diag(W.ss)*(W.V'*x));

if ~isempty(W.D)
  y=y+W.D*x;
end
