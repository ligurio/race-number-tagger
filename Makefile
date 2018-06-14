MODEL="data/model.h5"
CSV_RESULTS="data/mturk_results.csv"
ORIG_IMAGES_PATH="data/original_data/"
PROCESSED_IMAGES="data/race_numbers/"

TRAIN_RATIO=85
VALIDATE_RATIO=10
TEST_RATIO=5

PERCENTILE=75
BOX_HEIGHT?=166
BOX_WIDTH?=67

DATE=$(shell date +%Y-%m-%d)
BACKUP_FILE="race-number-tagger-$(DATE).tgz"
BASE=$(basename $PWD)

all: prepare train

test:
	python -m unittest test_bib

backup:
	@echo $(BASE)
	tar cvzf ../$(BACKUP_FILE) 				\
		--exclude='../$(BASE)/$(ORIG_IMAGES_PATH)/'	\
		--exclude='../$(BASE)/$(PROCESSED_IMAGES)/'	\
		--exclude='../$(BASE)/$(MODEL)'			\
		--exclude='../$(BASE)/$(BACKUP_FILE)' .
	@echo "Backup file - $(BACKUP_FILE)"

clean:
	rm -f *.pyc

review:
	python mturk-csv-review.py

prepare:
	python bib_prepare_dataset.py -csv $(CSV_RESULTS) 		\
			-percentile $(PERCENTILE) 			\
			-orig_images_dir $(ORIG_IMAGES_PATH)		\
			-processed_images_dir $(PROCESSED_IMAGES)	\
			-train_ratio $(TRAIN_RATIO)			\
			-validate_ratio $(VALIDATE_RATIO)		\
			-test_ratio $(TEST_RATIO)

train:
	python bib_train.py -box_w $(BOX_WIDTH)		\
			-box_h $(BOX_HEIGHT)		\
			-model_file $(MODEL)		\
			-processed_images_dir $(PROCESSED_IMAGES)
