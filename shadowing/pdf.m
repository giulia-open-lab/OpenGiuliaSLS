%==========================================================================
% Holger Claussen
% Bell Labs Research, Alcatel-Lucent
%==========================================================================

%--------------------------------------------------------------------------
% Function returns a vector with the cdf of a input data vector.
%--------------------------------------------------------------------------

function PDF = pdf(data, PDF_values)

PDF = zeros(size(PDF_values));
for i=1:size(PDF_values,2)-1
    PDF(i) = sum(sum((data > PDF_values(i)) & (data < PDF_values(i)+1)));
end
PDF = PDF./sum(PDF);  % normalize
