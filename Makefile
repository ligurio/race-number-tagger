all: test

test:
	python -m unittest test_bib

clean:
	rm -f *.pyc
