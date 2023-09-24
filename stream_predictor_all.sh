# RAITE 2023
TEAM_ID=Notre-Dame-2
MATCH_DURATION=7m20s
TIMESTAMP=$(date +'%Y-%m-%d_%H-%M')

# Configure before each match
#-------------------------------
MATCH_ID=01
CAM_LABEL=camera-20
RSTP_URL=rtsp://localhost:8554
#-------------------------------


MODELNAME=inverted_SINATRA
python stream_predictor_inverted_SINATRA_nix.py \
    --input_rtsp ${RSTP_URL}/${CAMLABEL}  \
    --output_rtsp ${RSTP_URL}/output__${MATCHID}__${TEAMID}__${CAMLABEL}__${MODELNAME}__${TIMESTAMP}  \
    --weights_path ${MODELNAME}_model.pth \
    --output_fname output/${MATCHID}__${TEAMID}__${CAMLABEL}__${MODELNAME}__${TIMESTAMP}.csv &

PID1=$!

MODELNAME=no_adv_training
python stream_predictor_x-entropy_nix.py \
    --input_rtsp ${RSTP_URL}/${CAMLABEL}  \
    --output_rtsp ${RSTP_URL}/output__${MATCHID}__${TEAMID}__${CAMLABEL}__${MODELNAME}__${TIMESTAMP}  \
    --weights_path ${MODELNAME}_model.pth \
    --output_fname output/${MATCHID}__${TEAMID}__${CAMLABEL}__${MODELNAME}__${TIMESTAMP}.csv & 

PID2=$!

# Sleep for match duration
sleep ${MATCH_DURATION}

# Kill the processes
kill -9 $PID1 $PID2