losses = {'sq','lr','sqw','hs'};

for ii=1:length(losses)
  loss=losses{ii};
  switch(loss)
   case 'sq'
    regs={'l1','gl','ds','en','l1n','al1'};
   case 'lr'
    regs={'l1','gl','ds','en','al1'};
   case 'sqw'
    regs={'l1'};
   case 'hs'
    regs={'gl'};
  end
  for jj=1:length(regs)
    reg=regs{jj};
    fname=sprintf('dal%s%s', loss, reg);
    fprintf('Testing %s...\n', fname);
    [~,out]=system(sprintf('cat %s.m', fname));
    out=split(out,char(10));
    ind1=find(cellfun(@(x)strcmp(x, '% Example:'),out));
    L=find(cellfun(@(x)strcmp(x, '%'), out(ind1+1:end)));
    for ll=1:L-1
      cmd=out{ind1+ll}(3:end);
      eval(cmd);
      ix=strfind(cmd,fname);
      if ~isempty(ix)
        wwogn=ww; statogn=stat;
        ixA=strfind(cmd, 'A');
        cmd2=[cmd(1:ixA-1), '{@(ww)A*ww, @(aa)A''*aa, m, prod(n)}', ...
              cmd(ixA+1:end)];
        fprintf('Testing %s with function handles...\n', fname);
        eval(cmd2);
      end
    end
  end
end