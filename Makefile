all: test

test:
	python -m unittest test_bib

backup:
	tar cvzf ../race-number-tagger-`date +%Y-%m-%d`.tgz --exclude='../race-number-tagger/data/original_data/*' --exclude='../race-number-tagger/data/race_numbers/*' --exclude='../race-number-tagger/data/*.h5' ../race-number-tagger

clean:
	rm -f *.pyc
