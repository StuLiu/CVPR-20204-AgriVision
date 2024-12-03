

export CUDA_VISIBLE_DEVICES=6,7

python main.py -c config/0527-FPN-efficientnet-b5-AcwBceJcd.yaml
python main.py -c config/0527-FPN-efficientnet-b4-AcwBceJcd.yaml
python main.py -c config/0529-DeepLabV3Plus-efficientnet-b5-Hybirdv4.yaml
python main.py -c config/0529-DeepLabV3Plus-efficientnet-b3-Hybirdv4.yaml
