#### GIN fine-tuning
for runseed in 0 1 2 3 4 5 6 7 8 9
do
python finetune.py --input_model_file PEM_DropH/pretrained.pth --split scaffold --runseed $runseed \
        --dataset "bace" --lr 1e-3 --epochs 100 --batch_size 64 
done
