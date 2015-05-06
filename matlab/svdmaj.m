function [U,S,V]=svdmaj(A, lambda, varargin)
opt=propertylist2struct(varargin{:});
opt=set_defaults(opt, 'kinit', 10, 'kstep', 2);

if isnumeric(A)
  MM=min(size(A));
else
  MM=min(A{3:4});
end

if isnumeric(A) && (max(size(A))<100 || opt.kinit==MM)
  [U,S,V]=svd(A);
else

  mm = inf;

  fprintf('[svdmaj]\n');

  kk=opt.kinit/opt.kstep;
  while mm>lambda && kk<MM
    kk=min(kk*opt.kstep, MM);
    fprintf('kk=%d\n',kk);
    if isnumeric(A)
      [U,S,V]=pca(A, kk, 10); % Using Mark Tygert's pca.m
    else
      [U,S,V]=lansvd(A{:},kk,'L',struct('s_target',lambda));
    end
    mm=min(diag(S));
  end
end