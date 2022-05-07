
accelerate launch train_detection_model.py \
  --train_file data/csv_file/trainall.times2.pkl.csv \
  --model_name_or_path ChineseBERT-base \
  --random_mask 5 \
  --batch_size 64 \
  --pad_to_max_length False \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --num_warmup_steps 1000 \
  --output_dir "model_output/ChineseBERT_ForCSC_Detection" \
  --seed 233 \
  --logging_steps 20 \
  --save_steps 10000 \
