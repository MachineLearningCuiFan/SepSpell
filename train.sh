

accelerate launch train_SepSpell.py \
  --train_file data/csv_file/trainall.times2.pkl.csv \
  --max_length 128  \
  --model_name_or_path ChineseBERT-base \
  --random_mask 5 \
  --batch_size 32 \
  --pad_to_max_length False \
  --learning_rate 5e-5 \
  --num_train_epochs 10 \
  --num_warmup_steps 5000 \
  --output_dir ChineseBERT_ForCSC_epoch10 \
  --seed 233 \
  --logging_steps 20 \
  --save_steps 10000 \
