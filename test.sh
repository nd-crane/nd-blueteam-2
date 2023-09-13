# pdm run python stream_recorder.py --streams stream-config.yaml --time_limit 35 & 
pdm run python stream_predictor.py --streams stream-config1.yaml --time_limit 30 & 
pdm run python stream_predictor.py --streams stream-config2.yaml --time_limit 30 &
# pdm run python stream_predictor.py --streams stream-config3.yaml --time_limit 30
# (trap 'kill 0' SIGINT; stream_recorder.py & stream_predictor.py)