% simula_LDA.m
clear all;
Nclas=3;
Ndim=10;
global N;

rand('seed', 100);
% Definir clases (gaussianas)
factor=1.0;
for n=1:Nclas
    mu{n} = rand(1,Ndim);
    Sig{n} = factor*genSymCov(Ndim);
end
medtot=zeros(1,Ndim);
for nclas=1:Nclas
    medtot=medtot+mu{nclas};
end
medtot=medtot/Nclas;

% Generar puntos
N=1000;
x=zeros(N,Ndim,Nclas);
for nclas=1:Nclas
    x(:,:,nclas)=mvnrnd(mu{nclas},Sig{nclas},N);
end
% figure(1)
% plot(x(:,1,1),x(:,2,1),'bx')
% axis([-0.5 3 -0.5 3])
% hold on
% plot(x(:,1,2),x(:,2,2),'g+')
% plot(x(:,1,3),x(:,2,3),'ro')
% hold off

% Computar ER original
disp('Error rate inicial')
ERini=getER(x,mu)

ncount=0;
for center=medtot-4:0.5:medtot+4
    ncount=ncount+1;
    % Transformar features
    [y,muT]=sigm(x,mu,center);
    ER(ncount)=getER(y,muT);
    % figure(2)
    % plot(y(:,1,1),y(:,2,1),'bx')
    % axis([-0.2 1.2 -0.2 1.2])
    % hold on
    % plot(y(:,1,2),y(:,2,2),'g+')
    % plot(y(:,1,3),y(:,2,3),'ro')
    % hold off
    % pause
    
    % Matrices covarianza intra- e inter-clase
    Sigw=zeros(Ndim,Ndim);
    Sigb=zeros(Ndim,Ndim);
    for nclas=1:Nclas
        data=y(:,:,nclas);
        Sigw=Sigw+cov(data);
        covtmp=mu{nclas}-medtot;
        Sigb=Sigb+covtmp*covtmp';
    end
    Sigw=Sigw/Nclas;
    Sigb=Sigb/Nclas;
    
    % Obtener LDA=A
    Siglda=inv(Sigw)*Sigb;
    [V,D]=eig(Siglda);
    lambda=diag(D);
    [lambda,I]=sort(lambda,'descend');
    A=(V(:,I))';
    Loss(ncount)=sum(lambda);
    Loss1(ncount)=sum(lambda(1:Nclas-1));
    Loss2(ncount)=sum(lambda(Nclas:end));
    
    % Obtener ER tras LDA
    ylda=y;
    for nclas=1:Nclas
        ytmp=A*y(:,:,nclas)';
        ylda(:,:,nclas)=ytmp';
        mutmp=A*muT{nclas}';
        muLDA{nclas}=mutmp';
    end
    ERlda(ncount)=getER(ylda,muLDA);

end

figure
center=medtot-4:0.5:medtot+4
subplot(1,4,1), plot(center,Loss)
title('Loss')
subplot(1,4,2), plot(center,Loss1)
title('Loss1')
subplot(1,4,3), plot(center,Loss2)
title('Loss2')
subplot(1,4,4), plot(center,ERlda)
title('ER (%)')

disp('Loss:')
Loss
disp('Loss1:')
Loss1
disp('Loss2:')
Loss2
disp('ER (%)')
ERlda


function [y,muT]=sigm(x,mu,center)
y=x;
muT=mu;
[Nfil,Ncol,Nclas]=size(x);
for nclas=1:Nclas
    for ncol=1:Ncol
        y(:,ncol,nclas)=1./(1+exp(-x(:,ncol,nclas)+center));
    end
    muT{nclas}=1./(1+exp(-mu{nclas}+center));
end
end


function ER=getER(x,mu)
global N
[Nfil,Ncol,Nclas]=size(x);
tablaconf=zeros(Nclas,Nclas);
for ntrue=1:Nclas
    for n=1:N
        dmin=1e10;
        for ntry=1:Nclas
            d=sum((x(n,:,ntrue)-mu{ntry}).^2);
            if (d<dmin)
                dmin=d;
                nrec=ntry;
            end
        end
        tablaconf(ntrue,nrec)=tablaconf(ntrue,nrec)+1;
    end
end
Ntot=Nclas*N;
ER=100*(Ntot-sum(diag(tablaconf)))/Ntot;
end

function cov=genSymCov(Ndim)
d = 100*rand(Ndim,1);
t = triu(bsxfun(@min,d,d.').*rand(Ndim),1);
cov = diag(d)+t+t.';
end

