function [ X ] = SemiSupervised_MUSIC( Y,phi,U,k,r,maxiter,epsilon )
%authors: Zaidao Wen 2017
% Solve min_{X} norm(Y-phi*X,'fro') s.t. ||X||_{row,0}=K with Semi-Supervised MUSIC
% Z. Wen, B. Hou and L. Jiao. "Joint Sparse Recovery with Semi-Supervised MUSIC" IEEE Signal Process. Lett.
%Input: Y: MMVs, phi: measurement matrix, U, Basis of Y (U\gets svds(Y,r)) or estimated from noisy MMVs, k: row sparsity, 
%       r: rank of pure signal subspace or estimated from noisy MMVs, maxiter: Maximum iterations, epsilon: permitted residuals
%Output: X: recovered row sparse signals.
%%init%%%%%%%%%%%%%%%%%%%%%%%%%%%
X=zeros(size(phi,2),size(Y,2));
PU=U*U';% projection operator onto S(Y)
I=eye(size(PU));
PU_orth=I-PU;% projection operator onto orthogonal complete of S(Y)
resi=zeros(1,maxiter);
%%%%%%%%main loop %%%%%%%%%%%%%%%%%%%%%%%%
iter=1;
while iter<maxiter
    % search k-r
   if iter==1
       U0=U;
   else
       [U0]=orth(R);% basis of feature domain
   end
    % matching
    Coh=phi'*U0;% NS classifier (simply version)
    rownorm=sum(Coh.^2,2);
    [~,index]=sort(rownorm,'descend');
    Supp1new=index(1:k-r);
    Supp1new=Supp1new(:);
    if iter==1
        Supp1=Supp1new;
    else
        Supptilde=[Supp(:);Supp1new(:)];
        Xp=X;
        Xp(Supptilde,:)=pinv(phi(:,Supptilde))*Y;
        rownorm=sum(Xp.^2,2);
        [~,indexnew]=sort(rownorm,'descend');
        Supp1=indexnew(1:k-r);% find k-r positive
        Supp1=Supp1(:);
    end
    % search rest r positive
    Urest=orth(PU_orth*phi(:,Supp1));
    Uall=[Urest,U];
    phinew=phi;
    phinew(:,Supp1)=0;
    Cohnew=phinew'*Uall; % NS classifier 
    rownorm=sum(Cohnew.^2,2);
    [~,index]=sort(rownorm,'descend');
    Supp2=index(1:r);
    Supp2=Supp2(:);
    % Check fitness
    Supp=[Supp2;Supp1];
    R=Y-phi(:,Supp)*(pinv(phi(:,Supp))*Y);
    resi(iter)=norm(R,'fro');
    if resi(iter)<epsilon
        break;
    else
        iter=iter+1;
    end
end
X(Supp,:)=pinv(phi(:,Supp))*Y;% output
end
% function T=union(T1,T2)
% L=length(T2);
% for i=1:L
%     if isempty(find(T1==T2(i), 1))
%         T1=[T1;T2(i)];
%     end
% end
% T=T1;
% end
