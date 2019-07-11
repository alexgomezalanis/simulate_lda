% simula_LDA.m
clear all;
Nclas=3;
Ndim=2;
global N;

% Definir clases (gaussianas)
factor=0.1;
mu{1}=[1,1];   Sig{1}=factor*[1 0; 0 1];
mu{2}=[2,1.5]; Sig{2}=factor*[1 0.5; 0.5 0.75];
mu{3}=[1.5,2]; Sig{3}=factor*[0.5 0.25;0.25 0.75];
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
figure(1)
plot(x(:,1,1),x(:,2,1),'bx')
axis([-0.5 3 -0.5 3])
hold on
plot(x(:,1,2),x(:,2,2),'g+')
plot(x(:,1,3),x(:,2,3),'ro')
hold off

% Computar ER original
disp('Error rate inicial')
ERini=getER(x,mu)

ncount=0;
for center=medtot-4:0.5:medtot+4
    ncount=ncount+1;
    % Transformar features
    [y,muT]=sigm(x,mu,center);
    ER(ncount)=getER(y,muT);
    figure(2)
    plot(y(:,1,1),y(:,2,1),'bx')
    axis([-0.2 1.2 -0.2 1.2])
    hold on
    plot(y(:,1,2),y(:,2,2),'g+')
    plot(y(:,1,3),y(:,2,3),'ro')
    hold off
    pause
    
    % Matrices covarianza intra- e inter-clase
    Sigw=zeros(Ndim,Ndim);
    for nclas=1:Nclas
        data=y(:,:,nclas);
        Sigw=Sigw+cov(data);
    end
    Sigw=Sigw/3.;
    Sigb=cov([muT{1};muT{2};muT{3}]);
    
    % Obtener LDA=A
    Siglda=inv(Sigw)*Sigb;
    [V,D]=eig(Siglda);
    lambda=diag(D);
    [lambda,I]=sort(lambda,'descend');
    A=(V(:,I))';
    Loss(ncount)=sum(lambda);
    Loss1(ncount)=lambda(1);
    Loss2(ncount)=lambda(2);
    
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

figure(3)
center=medtot-4:0.5:medtot+4
subplot(1,4,1), plot(center,Loss)
title('Loss')
subplot(1,4,2), plot(center,Loss1)
title('Loss1')
subplot(1,4,3), plot(center,Loss2)
title('Loss2')
subplot(1,4,4), plot(center,ERlda)
title('ER (%)')


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


