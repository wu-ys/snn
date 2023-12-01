n=0
for surrogate in atan erf piecewise_quad piecewise_exp soft_sign
do
  for alpha in 2.5 3.0 4.0 5.0
  do
    python3 plot.py -s $surrogate -a $alpha
  done
done