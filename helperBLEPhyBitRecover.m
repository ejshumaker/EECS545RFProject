function [cfgLLAdv,pktCnt,crcCnt,startIdx] = ...
                    helperBLEPhyBitRecover(rcv,prbIdx,pktCnt,crcCnt,bleParam)
%helperBLEPhyBitRecover Recovers the payload bits
%   [CFGLLADV,PKTCNT,CRCCNT,STARTIDX] =
%   helperBLEPhyBitRecover(RCV,PRBIDX,PKTCNT,CRCCNT,BLEPARAM) returns the
%   advertising channel PDU configuration object, CFGLLADV, from the
%   received BLE signal, RCV. STARTIDX is the start index for the
%   subsequent packet. Decoded packet statistics such as packet counter,
%   PKTCNT, and valid CRC counter, CRCCNT, are computed.
%
%   PKTCNT is the total number of detected BLE packets.
%
%   CRCCNT is the total number of BLE packets for which CRC passes.
%
%   BLEPARAM is a structure containing the BLE receiver parameters.
%
%   PRBIDX indicates the preamble index for the received BLE signal.
%
%   CFGLLADV represents the decoded PDU configuration, is an object of type
%   <a
%   href="matlab:help('bleLLAdvertisingChannelPDUConfig')">bleLLAdvertisingChannelPDUConfig</a>.
%
%   See also BLEReceiverExample, BLETransmitterExample.

%   Copyright 2019 The MathWorks, Inc.

refSeqLen = length(bleParam.RefSeq); % Reference sequence length
startIdx = min(length(rcv),2*bleParam.FrameLength); % Start index for the subsequent packet
cfgLLAdv = [];

if prbIdx >= refSeqLen
    syncFrame = rcv(1+prbIdx-refSeqLen:end); % Frame that always starts with a preamble

    if length(syncFrame) >= bleParam.MinimumPacketLen

        % GMSK demodulation
        demodBits = ble.internal.gmskdemod(syncFrame,bleParam.SamplesPerSymbol);

        % Received preamble
        rcvPreamble = demodBits(1:bleParam.PrbLen)>0;

        % Synchronized data
        demodSyncData = demodBits(1+bleParam.PrbLen:end);

        % Loop to recover the payload bits
        % A packet is considered as detected only when the received
        % preamble matches with the known preamble sequence
        if isequal(rcvPreamble,bleParam.Preamble)

             % Decode as per PHY mode
            if any(strcmp(bleParam.Mode,{'LE1M','LE2M'}))   % For LE1M or LE2M
                accAddress = int8(demodSyncData(1:bleParam.AccessAddLen)>0);
                decodeData = int8(demodSyncData(1+bleParam.AccessAddLen:end)>0);
            else                                            % For LE500K or LE125K
                if strcmp(bleParam.Mode,'LE500K') && (rem(length(demodSyncData),2) ~= 0)
                    padLen = 2 - rem(length(demodSyncData),2);
                elseif strcmp(bleParam.Mode,'LE125K') && (rem(length(demodSyncData),8) ~= 0)
                    padLen = 8 - rem(length(demodSyncData),8);
                else
                    padLen = 0;
                end
                demodSyncData = [demodSyncData; zeros(padLen,1)];
                [decodeData,accAddress] = ble.internal.decode(demodSyncData,bleParam.Mode);
            end

            % After packet detection, check for the access address, If the
            % access address does not match with the known access address
            % then the packet is considered as a lost packet.
            if isequal(accAddress,bleParam.AccessAddress)

                % Perform data dewhitening
                dewhitenStateLen = 6;
                chanIdxBin = comm.internal.utilities.de2biBase2LeftMSB(bleParam.ChannelIndex,dewhitenStateLen);
                initState = [1 chanIdxBin]; % Initial conditions of shift register
                dewhitenedBits = ble.internal.whiten(decodeData,initState);

                % Extract PDU length
                PDULenField = double(dewhitenedBits(9:16)); % Second byte of PDU header
                PDULenInBytes = comm.internal.utilities.bi2deRightMSB(PDULenField',2);
                PDULenInBits = PDULenInBytes*8;

                % Check for the length of dewhitenedBits. If the length of
                % dewhitenedBits is greater than or equal to (PDU length +
                % CRC length + Header length) then CRC check is performed.
                if length(dewhitenedBits) >= PDULenInBits+bleParam.CRCLength+bleParam.HeaderLength
                    pduBitsWithCRC = dewhitenedBits(1:bleParam.HeaderLength+PDULenInBits+bleParam.CRCLength);
                    [status, cfgLLAdv] = bleLLAdvertisingChannelPDUDecode(pduBitsWithCRC);
                    if strcmp(status, 'Success')
                        % Increment the CRC counter
                        crcCnt = crcCnt + 1;
                    end
                end
                startIdx = 2*bleParam.FrameLength-((length(dewhitenedBits)-PDULenInBits)*bleParam.SamplesPerSymbol); 
            end
            % Increment the packet counter
            pktCnt = pktCnt+1;
        end
    end
end