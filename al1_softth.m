% al1_softth - soft threshold function for adaptive L1 regularization
%
% Copyright(c) 2009- Ryota Tomioka, Satoshi Hara
% This software is distributed under the MIT license. See license.txt

function [vv,ss]=al1_softth(vv,pp,info)

n = size(vv,1);

Ip=find(vv>pp);
In=find(vv<-pp);

vv=sparse([Ip;In],1,[vv(Ip)-pp(Ip);vv(In)+pp(In)],n,1);

ss=abs(vv);