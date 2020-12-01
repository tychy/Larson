DATA = data
RESULT = results
test : module

module:
	python -m unittest test_modules.py
no_Q:mkdir
	python main_no_Q.py
mkdir:
	mkdir -p $(data)
	mkdir -p $(RESULT)