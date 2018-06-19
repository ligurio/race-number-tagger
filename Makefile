MODEL="data/model.h5"
CSV_RESULTS="data/mturk_results.csv"
ORIG_IMAGES_PATH="data/original_data/"
PROCESSED_IMAGES="data/race_numbers/"
TRAIN_LOG="train_log"

TRAIN_RATIO=85
VALIDATE_RATIO=10
TEST_RATIO=5
PERCENTILE=75

DATE=$(shell date +%Y-%m-%d)
BACKUP_FILE="race-number-tagger-$(DATE).tgz"
BASE=$(basename $PWD)

all: prepare train

test:
	python -m unittest test_bib

backup:
	@echo "Create backup $(BASE)"
	@tar cvzf ../$(BACKUP_FILE) 				\
		--exclude='../$(BASE)/$(ORIG_IMAGES_PATH)/'	\
		--exclude='../$(BASE)/$(PROCESSED_IMAGES)/'	\
		--exclude='../$(BASE)/$(MODEL)'			\
		--exclude='../$(BASE)/$(BACKUP_FILE)' .
	@echo "Backup file - $(BACKUP_FILE)"

clean:
	rm -f *.pyc $(MODEL) $(TRAIN_LOG)

review:
	@echo "Prepare images to review"
	python mturk-csv-review.py
	@echo "Build image list - find $(PROCESSED_IMAGES) -type f"

prepare:
	@echo "* Percentile:		$(PERCENTILE)"
	@echo "* Train ratio:		$(TRAIN_RATIO)"
	@echo "* Validate ratio:	$(VALIDATE_RATIO)"
	@echo "* Test ratio:		$(TEST_RATIO)"
	@echo "* Processed image dir:	$(PROCESSED_IMAGES)"
	@echo "* Original image dir:	$(ORIG_IMAGES_PATH)"
	@echo
	@echo "Preparing dataset..."
	@python bib_prepare_dataset.py --csv $(CSV_RESULTS) 		\
			--percentile $(PERCENTILE) 			\
			--orig_images_dir $(ORIG_IMAGES_PATH)		\
			--processed_images_dir $(PROCESSED_IMAGES)	\
			--train_ratio $(TRAIN_RATIO)			\
			--validate_ratio $(VALIDATE_RATIO)		\
			--test_ratio $(TEST_RATIO)

train:
ifeq ($(strip $(BOX_HEIGHT)),)
	@echo "Please set both BOX_HEIGHT and BOX_WIDTH variables"
	@echo "Calculating box size..."
	@python bib_prepare_dataset.py --csv $(CSV_RESULTS)		\
				--percentile $(PERCENTILE)		\
				--calc_size
else
	@echo "* Box width:		$(BOX_WIDTH)"
	@echo "* Box height:		$(BOX_HEIGHT)"
	@echo "* Processed image dir:	$(PROCESSED_IMAGES)"
	@echo
	@python bib_train.py --processed_images_dir $(PROCESSED_IMAGES)	\
			--box_w $(BOX_WIDTH) --box_h $(BOX_HEIGHT)
endif

update-modules:
	@echo "Update Python modules"
	@pip install --upgrade pip
	@pip freeze --local | grep -v '^\-e' | cut -d = -f 1  | xargs -n1 pip install -U
