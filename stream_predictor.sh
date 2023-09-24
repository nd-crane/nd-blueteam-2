# how to use examples

pdm run python stream_predictor.py --stream_url rtsp://admin:Marialufy2@192.168.0.101:65534 --weights_path models/best_model.pth --time_limit 600 --output_folder output-redteam-0

pdm run python stream_predictor_thread.py --stream_url rtsp://192.168.0.103:554/H264Video  --weights_path models/best_model.pth --time_lim
it 600 --output_folder output-redteam-0


pdm run python stream_predictor_x-entropy.py --stream_url rtsp://localhost:8554/camera-20  --weights_path models/best_model.pth --time_limit 600 --output_folder output-redteam-0