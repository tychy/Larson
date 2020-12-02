DATA = data
RESULT = results
test : module

module:
	python -m unittest test_modules.py
no_Q:mkdir
	python main_no_Q.py
plot:mkdir
	python plot.py
mkdir:
	mkdir -p $(DATA)
	mkdir -p $(RESULT)