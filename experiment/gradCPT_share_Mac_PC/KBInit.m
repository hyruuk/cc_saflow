function [KB_id] = KBInit()

KB_id=[];

% List of vendor IDs for valid FORP devices:
vendorIDs = [1452];

KeyPressed  =   '';
keydata     =   [];

% Try to detect first FORP device at first invocation:
if isempty(KB_id)
    Devices = PsychHID('Devices');
    % Loop through all KEYBOARD devices with the vendorID of FORP's vendor:
    for i=1:size(Devices,2)
        if strcmp(Devices(i).usageName,'Keyboard') & ismember(Devices(i).vendorID, vendorIDs)
            KB_id=i;
            break;
        end
    end
end

if isempty(KB_id)
    error('No Keyboard-Device detected on your system');
end