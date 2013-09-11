function kl = CalcKLdiv(q,p,xmin,xmax,ymin,ymax)

kl = dblquad(@(x1,x2)klfun(x1,x2,q,p),xmin,xmax,ymin,ymax);

end

function dr = klfun(x1,x2,q,p)

dr = zeros(size(x1));

qdens = q(x1,x2);
pdens = p(x1,x2);
sel=(qdens>1e-5 & pdens>1e-8);

dr(sel) = qdens(sel).*(log(qdens(sel))-log(pdens(sel)));

end