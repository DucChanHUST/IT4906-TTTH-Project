for i in {0..9}
do
  dataset="100_$i"
  python3 moead_exp.py $dataset 
  python3 nsga_exp.py $dataset
done

for i in {0..9}
do
  dataset="150_$i"
  python3 moead_exp.py $dataset
  python3 nsga_exp.py $dataset
done

for i in {0..9}
do
  dataset="200_$i"
  python3 moead_exp.py $dataset
  python3 nsga_exp.py $dataset
done

for i in {0..9}
do
  dataset="250_$i"
  python3 moead_exp.py $dataset
  python3 nsga_exp.py $dataset
done