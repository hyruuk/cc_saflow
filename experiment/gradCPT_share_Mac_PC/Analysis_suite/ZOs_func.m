% The matrix variable "response" is fed into the function ZOs() with the
% resultant output assigned to ZO_count
% inputs 38,.01 will look for no response in 30 secs

function [ZO_count] = ZOs_func(response,window,threshold)

% 10 trials
%window = 10;

% This threshold is set to when the subject fails to press (,) during a
% city trial, 80% of the time.
%threshold = .4; 

ZO_count = 0;

for x=1:(size(response,1)-window)

    response_temp=response(x:x+window-1,:);
    
% CC is comission correct: a trial where the press (,) was for a city.
%
% Find instances where column 1 denoting 2 (where city is displayed) overlaps
% with the 7th colum denoting 1 (pressing of , for city).
% 
% find(response_temp(:,1)==2 & response_temp(:,7)==1)
%
% Then, count the number of correct comission trials using size().
% Note: the 1 in size(variable,1) indicates number of rows, not colums
% (where the value would be 2).

    CC=size(find(response_temp(:,1)==2 & response_temp(:,7)==1),1);
    
% OE is omission error: a trial where the user makes an error by omitting a
% response.
% 
% Find instances where column 1 is denoting 2 (where city is displayed) but
% column 7 is 0 (indicating that the user did not press anything; there was
% no response).
%
% find(response_temp(:,1)==2 & response_temp(:,7)==0)
%
% Then, count the number of errors in omission using size().

    OE=size(find(response_temp(:,1)==2 & response_temp(:,7)==0),1);

% Add the correct comission (, pressed for city) and errors in omission (,
% pressed for mountain). Divide the correct comission by the total presses
% (,) and if it's less than or equal to the threshold, increment ZO_count
% by 1.
%
% For example, if CC/(CC+OE) is less than 0.2, then that means that the
% total correct (,) presses for the city was less than the total presses
% (,) attempted.
    
    if CC/(CC+OE)<=threshold
        ZO_count = ZO_count+1;    
    end;

end;

%omissions

omission_errors = size(find(response(:,1)==2 & response(:,7)==0),1);

correct_response = size(find(response(:,1)==2 & response(:,7)==1),1);

omission_rate=omission_errors/(correct_response+omission_errors);

%['ZO count= ' num2str(ZO_count) ' OE= ' num2str(omission_rate)]
