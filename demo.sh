name="box"

python main.py $name	${name}_manipulated_moved.obj	result_of_${name}	5	600	5	600	5	5	600	0.5	0	cuda:0
python main.py $name	noised_${name}_manipulated_moved.obj	result_of_noised_${name}	2	600	1	300	8	1	20	0.5	1	cuda:0
python main.py $name	scanned_${name}_iphone11_low_100mm_aligned.obj	result_of_scanned_${name}	2	600	1	100	5	1	50	0.5	1	cuda:0


