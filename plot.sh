n=0
for surrogate in atan erf piecewise_quad piecewise_exp soft_sign
do
  for alpha in 0.25 0.75 1.5
  do
    python3 plot.py -s $surrogate -a $alpha
  done
done