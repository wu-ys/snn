n=0
CUDA_VISIBLE_DEVICES=2
for surrogate in atan erf piecewise_quad piecewise_exp soft_sign
do
  for alpha in 2.5 3.0 4.0 5.0
  do
    nohup python3 snn.py -s $surrogate -a $alpha > nohup/nohup_"$surrogate"_"$alpha".out 2>&1 &
  done
done

