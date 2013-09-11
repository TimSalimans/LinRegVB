function r2 = CalcRsquared(q,p,xmin,xmax,ymin,ymax)

kl2 = dblquad(@(x1,x2)klfun2(x1,x2,q,p),xmin,xmax,ymin,ymax);
kl3 = dblquad(@(x1,x2)klfun3(x1,x2,q,p),xmin,xmax,ymin,ymax);
kl4 = dblquad(@(x1,x2)klfun4(x1,x2,q,p),xmin,xmax,ymin,ymax);

r2=1-kl2./(kl4-kl3.^2);

end

function dr = klfun2(x1,x2,q,p)

dr = zeros(size(x1));

qdens = q(x1,x2);
pdens = p(x1,x2);
sel=(qdens>1e-5 & pdens>1e-8);

dr(sel) = qdens(sel).*(log(qdens(sel))-log(pdens(sel))).^2;
end

function dr = klfun3(x1,x2,q,p)

dr = zeros(size(x1));

qdens = q(x1,x2);
pdens = p(x1,x2);
sel=(qdens>1e-5 & pdens>1e-8);

dr(sel) = qdens(sel).*log(pdens(sel));
end

function dr = klfun4(x1,x2,q,p)

dr = zeros(size(x1));

qdens = q(x1,x2);
pdens = p(x1,x2);
sel=(qdens>1e-5 & pdens>1e-8);

dr(sel) = qdens(sel).*log(pdens(sel)).^2;
end