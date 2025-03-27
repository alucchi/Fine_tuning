# Fine_tuning
Code to fine-tune LLMs
The example is to fine-tune a pretrained LLM (Llama, Qwen, etc)

How to run it?
python3 SFT_finetuning.py --batch_size 4 --optim_lr 1e-4 --n_training 300 --optim_name signadam
