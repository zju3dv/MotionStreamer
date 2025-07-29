export HF_ENDPOINT=https://hf-mirror.com
cd Evaluator_272
huggingface-cli download --resume-download distilbert/distilbert-base-uncased --local-dir ./deps/distilbert-base-uncased
ln -s ../humanml3d_272 ./datasets/humanml3d_272
python -m train --cfg configs/configs_evaluator_272/H3D-TMR.yaml --cfg_assets configs/assets.yaml --batch_size 256 --nodebug
cd ..