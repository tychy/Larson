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

adiabatic:mkdir
	python main_adiabatic.py
	python plot.py
	python plot_core.py

ionization:mkdir
	python main_ionization.py
	python plot.py
	python plot_core.py
	python plot_ion.py

plot:mkdir
	python plot.py

plot_core:mkdir
	python plot.py
	python plot_core.py
plot_ion:mkdir
	python plot.py
	python plot_core.py
	python plot_ion.py

mkdir:
	mkdir -p $(DATA)
	mkdir -p $(RESULT)
