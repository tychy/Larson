DATA = data
RESULT = results
test : module

module:
	python -m unittest test_modules.py
no_Q:mkdir
	python main_no_Q.py
	python plot.py

with_Q:mkdir
	python main_with_Q.py
	python plot.py

energy:mkdir
	python main_energy.py
	python plot.py
	python plot_core.py

plot:mkdir
	python plot.py
mkdir:
	mkdir -p $(DATA)
	mkdir -p $(RESULT)