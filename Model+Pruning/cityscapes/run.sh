mkdir -p ./save
mkdir -p ./trainlogs
method=famo
seed=42
gamma=0.01
python trainer.py --method=$method --seed=$seed --gamma=$gamma &> "trainlogs/famo-gamma.log" &